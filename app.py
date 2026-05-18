"""
app.py - Smart Academic Advisor Flask app.
Run: python app.py
Open: http://127.0.0.1:5000
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from sklearn.metrics import pairwise_distances

app = Flask(__name__)
try:
    with open("models/trained_models.pkl", "rb") as f:
        MODEL_DATA = pickle.load(f)
    print("[OK] Models loaded successfully.")
except FileNotFoundError:
    print("[WARN] No trained models found. Run python setup.py first.")
    MODEL_DATA = None

COURSE_KEYS = ["math", "programming", "data_structures", "algorithms", "databases", "networks", "ai", "web"]
GRADE_FEATURES = [f"{key}_grade" for key in COURSE_KEYS]

COURSE_LABELS = {
    "math_grade": "Mathematics",
    "programming_grade": "Programming",
    "data_structures_grade": "Data Structures",
    "algorithms_grade": "Algorithms",
    "databases_grade": "Databases",
    "networks_grade": "Networks",
    "ai_grade": "Artificial Intelligence",
    "web_grade": "Web Development",
}

TRACK_META = {
    "AI/Machine Learning": {
        "short": "AI/ML",
        "icon": "AI",
        "color": "#7c3aed",
        "bg": "#f3e8ff",
        "interest": "AI/ML",
        "core": ["math_grade", "programming_grade", "data_structures_grade", "algorithms_grade", "ai_grade"],
        "electives": ["Machine Learning", "Deep Learning", "Computer Vision", "Natural Language Processing", "Reinforcement Learning"],
        "careers": ["Machine Learning Engineer", "AI Research Assistant", "Computer Vision Engineer", "NLP Engineer"],
        "desc": "Best for students who enjoy intelligent systems, data-driven models, algorithms, and advanced AI applications.",
    },
    "Web Development": {
        "short": "Web",
        "icon": "WEB",
        "color": "#059669",
        "bg": "#dcfce7",
        "interest": "Web Development",
        "core": ["programming_grade", "data_structures_grade", "databases_grade", "web_grade"],
        "electives": ["Advanced Front-End Development", "Back-End APIs", "UI/UX Design", "Cloud Deployment", "Web Security"],
        "careers": ["Full-Stack Developer", "Front-End Engineer", "Back-End Developer", "Web Application Developer"],
        "desc": "Best for students who want to build modern, usable, and scalable websites and web applications.",
    },
    "Network Engineering": {
        "short": "Networks",
        "icon": "NET",
        "color": "#d97706",
        "bg": "#fef3c7",
        "interest": "Networks",
        "core": ["programming_grade", "networks_grade", "math_grade"],
        "electives": ["Advanced Networking", "Cybersecurity", "Cloud Computing", "Network Administration", "Ethical Hacking"],
        "careers": ["Network Engineer", "Security Analyst", "Cloud Infrastructure Engineer", "Systems Administrator"],
        "desc": "Best for students who like infrastructure, communication systems, security, and cloud/network operations.",
    },
    "Data Science": {
        "short": "Data",
        "icon": "DATA",
        "color": "#2563eb",
        "bg": "#dbeafe",
        "interest": "Data Science",
        "core": ["math_grade", "programming_grade", "algorithms_grade", "databases_grade", "ai_grade"],
        "electives": ["Statistics and Probability", "Data Mining", "Big Data Analytics", "Data Visualization", "Business Intelligence"],
        "careers": ["Data Scientist", "Data Analyst", "BI Developer", "Analytics Engineer"],
        "desc": "Best for students who enjoy statistics, databases, patterns, dashboards, and decision-making from data.",
    },
    "Software Engineering": {
        "short": "Software",
        "icon": "SE",
        "color": "#dc2626",
        "bg": "#fee2e2",
        "interest": "Software Engineering",
        "core": ["programming_grade", "data_structures_grade", "algorithms_grade", "databases_grade", "web_grade"],
        "electives": ["Software Architecture", "Design Patterns", "Agile Development", "DevOps and CI/CD", "Software Testing and QA"],
        "careers": ["Software Engineer", "System Architect", "DevOps Engineer", "QA Automation Engineer"],
        "desc": "Best for students who want to design reliable, maintainable software systems using engineering practices.",
    },
}

INTEREST_OPTIONS = ["AI/ML", "Web Development", "Networks", "Data Science", "Software Engineering", "Undecided"]
INTEREST_TO_TRACK = {meta["interest"]: track for track, meta in TRACK_META.items()}
CLASS_SHORT_LABELS = {track: meta["short"] for track, meta in TRACK_META.items()}
ALGORITHM_LABELS = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbors",
}


def clamp_int(value: str, default: int = 70, low: int = 0, high: int = 100) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def clamp_float(value: str, default: float = 0.0, low: float = 0.0, high: float = 4.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def auto_gpa(grades: Dict[str, int]) -> float:
    values = list(grades.values())
    if not values:
        return 0.0
    return round(float(np.mean(values) / 25), 2)


def build_feature_row(grades: Dict[str, int], gpa: float, interest: str) -> pd.DataFrame:
    row = {"gpa": gpa, "interest": interest or "Undecided"}
    row.update(grades)
    passed_count = sum(1 for value in grades.values() if value >= 60)
    failed_count = sum(1 for value in grades.values() if value < 60)
    row["passed_count"] = passed_count
    row["failed_count"] = failed_count
    row["excellent_count"] = sum(1 for value in grades.values() if value >= 85)
    row["weak_count"] = sum(1 for value in grades.values() if value < 70)
    for key in COURSE_KEYS:
        grade = grades[f"{key}_grade"]
        row[f"{key}_passed"] = int(grade >= 60)
        row[f"{key}_failed"] = int(grade < 60)
    columns = MODEL_DATA.get("features", list(row.keys())) if MODEL_DATA else list(row.keys())
    return pd.DataFrame([{col: row.get(col, 0) for col in columns}])


def grade_level(value: int) -> str:
    if value >= 85:
        return "excellent"
    if value >= 70:
        return "good"
    if value >= 60:
        return "pass"
    return "failed"


def get_student_items(grades: Dict[str, int], interest: str) -> set[str]:
    labels = {
        "math_grade": "Math", "programming_grade": "Programming", "data_structures_grade": "DataStructures",
        "algorithms_grade": "Algorithms", "databases_grade": "Databases", "networks_grade": "Networks",
        "ai_grade": "AI", "web_grade": "Web",
    }
    items = set()
    for col, label in labels.items():
        value = grades.get(col, 0)
        if value >= 85:
            items.add(f"Excellent_{label}")
        elif value >= 70:
            items.add(f"Good_{label}")
        elif value < 60:
            items.add(f"Failed_{label}")
        else:
            items.add(f"Weak_{label}")
    safe_interest = (interest or "Undecided").replace("/", "_").replace(" ", "_")
    items.add(f"Interest_{safe_interest}")
    items.add("Has_Failed_Course" if any(v < 60 for v in grades.values()) else "No_Failed_Courses")
    return items


def get_matching_rules(grades: Dict[str, int], interest: str) -> List[Dict[str, object]]:
    if MODEL_DATA is None:
        return []
    rules_df = MODEL_DATA.get("association_rules")
    if rules_df is None or len(rules_df) == 0:
        return []
    student_items = get_student_items(grades, interest)
    matched = []
    for _, row in rules_df.iterrows():
        antecedents = set(row["antecedents"])
        consequents = set(row["consequents"])
        if antecedents.issubset(student_items) and any(str(c).startswith("Track_") for c in consequents):
            matched.append({
                "antecedents": sorted(antecedents),
                "consequents": sorted(consequents),
                "confidence": round(float(row["confidence"]) * 100, 1),
                "support": round(float(row["support"]) * 100, 1),
                "lift": round(float(row["lift"]), 2),
            })
    return matched[:5]


def build_explanation(track: str, grades: Dict[str, int], interest: str, track_probs: Dict[str, float], rules: List[Dict[str, object]]) -> List[str]:
    meta = TRACK_META[track]
    reasons = []
    if interest == meta["interest"]:
        reasons.append(f"Your selected interest matches the {track} track.")
    elif interest and interest != "Undecided":
        interest_track = INTEREST_TO_TRACK.get(interest)
        if interest_track and interest_track != track:
            reasons.append(f"Your interest points toward {interest_track}, but your grades more strongly support {track}.")
    else:
        reasons.append("No specific interest was selected, so the model relied more on your academic performance.")

    core_grades = [(COURSE_LABELS[c], grades[c]) for c in meta["core"]]
    strong_core = [name for name, value in core_grades if value >= 80]
    if strong_core:
        reasons.append("Strong core performance in: " + ", ".join(strong_core[:4]) + ".")
    weak_core = [name for name, value in core_grades if value < 70]
    if weak_core:
        reasons.append("Some core subjects need attention: " + ", ".join(weak_core[:3]) + ".")

    top_prob = track_probs.get(track, 0)
    reasons.append(f"The selected model gave {track} the highest confidence score at {top_prob}%.")
    if rules:
        reasons.append(f"Association rules found historical student patterns supporting this recommendation with up to {rules[0]['confidence']}% confidence.")
    return reasons


def build_advising_warnings(track: str, grades: Dict[str, int]) -> List[str]:
    warnings = []
    failed = [COURSE_LABELS[key] for key, value in grades.items() if value < 60]
    if failed:
        warnings.append("Retake or improve failed courses before taking advanced electives: " + ", ".join(failed) + ".")

    core = TRACK_META[track]["core"]
    weak_core = [COURSE_LABELS[key] for key in core if grades[key] < 70]
    if weak_core:
        warnings.append("For this track, strengthen these prerequisite areas first: " + ", ".join(weak_core) + ".")

    if track in {"AI/Machine Learning", "Data Science"} and grades["math_grade"] < 70:
        warnings.append("Mathematics is important for this path. Add statistics, linear algebra, or probability practice.")
    if track in {"Software Engineering", "Web Development"} and grades["programming_grade"] < 70:
        warnings.append("Programming is a key prerequisite. Complete extra coding practice before advanced software courses.")
    if track == "Network Engineering" and grades["networks_grade"] < 70:
        warnings.append("Networks is the main foundation course for this path. Retake or review it before advanced networking/security electives.")
    if not warnings:
        warnings.append("No critical risk detected. You can start track electives while keeping your GPA stable.")
    return warnings


def build_graduation_path(track: str, grades: Dict[str, int]) -> List[Dict[str, object]]:
    meta = TRACK_META[track]
    failed_courses = [COURSE_LABELS[key] for key, value in grades.items() if value < 60]
    weak_core = [COURSE_LABELS[key] for key in meta["core"] if 60 <= grades[key] < 70]
    path = []
    first_actions = []
    if failed_courses:
        first_actions.append("Retake: " + ", ".join(failed_courses[:3]))
    if weak_core:
        first_actions.append("Review: " + ", ".join(weak_core[:3]))
    if not first_actions:
        first_actions.append("Start the first two recommended electives")
    first_actions.append("Meet advisor to confirm prerequisite order")
    path.append({"term": "Next semester", "items": first_actions})
    path.append({"term": "Following semester", "items": meta["electives"][:3] + ["Portfolio or mini-project in " + track]})
    path.append({"term": "Graduation preparation", "items": [meta["electives"][3], meta["electives"][4], "Capstone project aligned with " + track, "Internship or professional certification"]})
    return path


def safe_display_text(value: object, default: str = "None") -> str:
    """Return clean text for report fields that may arrive as NaN/empty values."""
    if pd.isna(value):
        return default
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null"}:
        return default
    return text


def find_similar_students(student_df: pd.DataFrame, limit: int = 4) -> List[Dict[str, object]]:
    if MODEL_DATA is None or "knn" not in MODEL_DATA.get("models", {}):
        return []
    try:
        knn_model = MODEL_DATA["models"]["knn"]
        preprocessor = knn_model.named_steps["preprocessor"]
        train_features = MODEL_DATA["training_features"]
        historical = MODEL_DATA["historical_students"]
        train_matrix = preprocessor.transform(train_features)
        student_matrix = preprocessor.transform(student_df)
        distances = pairwise_distances(student_matrix, train_matrix, metric="euclidean")[0]
        indices = np.argsort(distances)[:limit]
        similar = []
        max_distance = max(float(distances[indices[-1]]), 0.001)
        for idx in indices:
            row = historical.iloc[int(idx)]
            failed_courses = safe_display_text(row.get("failed_courses", "None"))
            similar.append({
                "student_id": safe_display_text(row["student_id"], "Unknown"),
                "gpa": round(float(row["gpa"]), 2),
                "interest": safe_display_text(row["interest"], "Undecided"),
                "track": safe_display_text(row["recommended_track"], "Unknown"),
                "failed_courses": failed_courses,
                "similarity": round(max(0.0, 100 - (float(distances[idx]) / max_distance) * 25), 1),
            })
        return similar
    except Exception as exc:
        print(f"[WARN] Similar student lookup failed: {exc}")
        return []


def get_prediction(model, feature_row: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    label_encoder = MODEL_DATA["label_encoder"]
    pred_encoded = model.predict(feature_row)[0]
    track = label_encoder.inverse_transform([int(pred_encoded)])[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_row)[0]
        class_ids = model.classes_
        probs = {
            label_encoder.inverse_transform([int(class_id)])[0]: round(float(prob) * 100, 1)
            for class_id, prob in zip(class_ids, probabilities)
        }
    else:
        probs = {track: 100.0}
    for class_name in MODEL_DATA.get("classes", []):
        probs.setdefault(class_name, 0.0)
    return track, dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))


def compare_all_models(feature_row: pd.DataFrame, selected_track: str) -> List[Dict[str, object]]:
    """Run the same student profile through every trained model for a clear demo comparison."""
    comparison = []
    if MODEL_DATA is None:
        return comparison
    for key, model in MODEL_DATA.get("models", {}).items():
        track, probs = get_prediction(model, feature_row)
        comparison.append({
            "key": key,
            "algorithm": ALGORITHM_LABELS.get(key, key.replace("_", " ").title()),
            "track": track,
            "confidence": probs.get(track, 0.0),
            "agrees": track == selected_track,
            "accuracy": MODEL_DATA.get("accuracies", {}).get(key, 0),
        })
    return comparison


@app.route("/")
def index():
    if MODEL_DATA is None:
        return render_template("setup_needed.html")
    return render_template("index.html", algorithm_labels=ALGORITHM_LABELS, interests=INTEREST_OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL_DATA is None:
        return redirect(url_for("index"))

    algorithm = request.form.get("algorithm", "random_forest")
    if algorithm not in MODEL_DATA.get("models", {}):
        algorithm = "random_forest"

    interest = request.form.get("interest", "Undecided")
    if interest not in INTEREST_OPTIONS:
        interest = "Undecided"

    grades = {feature: clamp_int(request.form.get(feature, "70")) for feature in GRADE_FEATURES}
    entered_gpa = clamp_float(request.form.get("gpa", "0"))
    gpa = entered_gpa if entered_gpa > 0 else auto_gpa(grades)

    feature_row = build_feature_row(grades, gpa, interest)
    model = MODEL_DATA["models"][algorithm]
    track, track_probs = get_prediction(model, feature_row)
    model_comparison = compare_all_models(feature_row, track)
    failed = [COURSE_LABELS[key] for key, value in grades.items() if value < 60]
    rules = get_matching_rules(grades, interest)
    explanation = build_explanation(track, grades, interest, track_probs, rules)
    warnings = build_advising_warnings(track, grades)
    graduation_path = build_graduation_path(track, grades)
    similar_students = find_similar_students(feature_row)

    return render_template(
        "result.html",
        track=track,
        meta=TRACK_META[track],
        track_probs=track_probs,
        algorithm=algorithm,
        algorithm_label=ALGORITHM_LABELS.get(algorithm, algorithm),
        grades=grades,
        grade_levels={key: grade_level(value) for key, value in grades.items()},
        gpa=gpa,
        interest=interest,
        failed=failed,
        rules=rules,
        explanation=explanation,
        warnings=warnings,
        graduation_path=graduation_path,
        similar_students=similar_students,
        model_comparison=model_comparison,
        course_labels=COURSE_LABELS,
        all_metas=TRACK_META,
        accuracy=MODEL_DATA["accuracies"].get(algorithm, 0),
    )




def load_dashboard_dataset() -> pd.DataFrame:
    """Dashboard and model evidence use the generated synthetic dataset directly."""
    path = "data/students.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@app.route("/dashboard")
def dashboard():
    if MODEL_DATA is None:
        return redirect(url_for("index"))

    df = load_dashboard_dataset()
    track_counts = df["recommended_track"].value_counts().to_dict()
    interest_counts = df["interest"].value_counts().to_dict()
    avg_gpa_by_track = df.groupby("recommended_track")["gpa"].mean().round(2).to_dict()
    avg_grades = df[GRADE_FEATURES].mean().round(1).to_dict()
    failed_rate = round(float((df["failed_count"] > 0).mean()) * 100, 1)

    top_rules = []
    rules_df = MODEL_DATA.get("association_rules")
    if rules_df is not None:
        for _, row in rules_df.head(12).iterrows():
            top_rules.append({
                "antecedents": row["antecedents"],
                "consequents": row["consequents"],
                "support": round(float(row["support"]) * 100, 1),
                "confidence": round(float(row["confidence"]) * 100, 1),
                "lift": round(float(row["lift"]), 2),
            })

    return render_template(
        "dashboard.html",
        total_students=len(df),
        failed_rate=failed_rate,
        track_counts=track_counts,
        interest_counts=interest_counts,
        avg_gpa=avg_gpa_by_track,
        avg_grades={COURSE_LABELS[key]: value for key, value in avg_grades.items()},
        accuracies=MODEL_DATA["accuracies"],
        cv_scores=MODEL_DATA["cv_scores"],
        fi=MODEL_DATA.get("feature_importance", {}),
        reports=MODEL_DATA["reports"],
        classes=MODEL_DATA["classes"],
        class_short_labels=[CLASS_SHORT_LABELS.get(name, name) for name in MODEL_DATA["classes"]],
        cms=MODEL_DATA["confusion_matrices"],
        top_rules=top_rules,
        track_metas=TRACK_META,
        algorithm_labels=ALGORITHM_LABELS,
    )


if __name__ == "__main__":
    debug_enabled = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_enabled, port=5000)
