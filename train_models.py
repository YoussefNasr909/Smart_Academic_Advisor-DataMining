"""
train_models.py
Trains Decision Tree, Random Forest, KNN, and mines Association Rules.
Run: python train_models.py
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# These are the direct course-grade inputs used by the classifiers.
# بالمصري: دي درجات المواد اللي الموديلات بتعتمد عليها في تحديد التراك.
GRADE_FEATURES = [
    "math_grade", "programming_grade", "data_structures_grade", "algorithms_grade",
    "databases_grade", "networks_grade", "ai_grade", "web_grade",
]
# These summary features describe the student's overall academic status.
# بالمصري: دي ملخصات مهمة زي عدد المواد الساقطة أو الممتازة.
SUMMARY_FEATURES = ["gpa", "passed_count", "failed_count", "excellent_count", "weak_count"]
BINARY_FEATURES = []
for course in ["math", "programming", "data_structures", "algorithms", "databases", "networks", "ai", "web"]:
    BINARY_FEATURES.extend([f"{course}_passed", f"{course}_failed"])
# Numeric features include grades, counts, GPA, and pass/fail indicators.
# بالمصري: كل دول أرقام، فهيدخلوا في جزء الـ scaling.
NUMERIC_FEATURES = ["gpa", *GRADE_FEATURES, "passed_count", "failed_count", "excellent_count", "weak_count", *BINARY_FEATURES]
CATEGORICAL_FEATURES = ["interest"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "recommended_track"

# These labels make Apriori rule items easier to read.
# بالمصري: دي أسماء مختصرة عشان قواعد association rules تبقى مقروءة.
GRADE_LABELS = {
    "math_grade": "Math", "programming_grade": "Programming", "data_structures_grade": "DataStructures",
    "algorithms_grade": "Algorithms", "databases_grade": "Databases", "networks_grade": "Networks",
    "ai_grade": "AI", "web_grade": "Web",
}


# This function creates the encoder that turns student interests into numeric columns.
# بالمصري: الموديل مبيفهمش text، فلازم نحول interest لأعمدة رقمية.
def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# This function prepares numeric and categorical features before model training.
# بالمصري: هنا بنعمل scaling للأرقام وencoding للاهتمام قبل التدريب.
def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# This function combines preprocessing and a classifier into one reusable model pipeline.
# بالمصري: كده بنضمن إن نفس preprocessing يتطبق في التدريب والتوقع.
def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", classifier),
    ])


# This function converts student records into items for Apriori association rules.
# بالمصري: Apriori محتاج items زي Good_AI مش أرقام خام.
def build_transactions(df: pd.DataFrame) -> list[list[str]]:
    transactions = []
    for _, row in df.iterrows():
        items = []
        for col, label in GRADE_LABELS.items():
            g = row[col]
            if g >= 85:
                items.append(f"Excellent_{label}")
            elif g >= 70:
                items.append(f"Good_{label}")
            elif g < 60:
                items.append(f"Failed_{label}")
            else:
                items.append(f"Weak_{label}")

        interest = str(row["interest"]).replace("/", "_").replace(" ", "_")
        items.append(f"Interest_{interest}")
        if int(row.get("failed_count", 0)) > 0:
            items.append("Has_Failed_Course")
        else:
            items.append("No_Failed_Courses")
        track = str(row[TARGET]).replace("/", "_").replace(" ", "_")
        items.append(f"Track_{track}")
        transactions.append(items)
    return transactions


# This function groups Random Forest feature importance into labels that are easy to explain.
# بالمصري: بدل ما نعرض أسماء أعمدة كتير، بنجمعها في عناوين سهلة للداشبورد.
def cleaned_feature_importance(model: Pipeline) -> dict[str, float]:
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return {}
    preprocessor = model.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    grouped = defaultdict(float)
    for raw_name, importance in zip(names, importances):
        name = raw_name.replace("num__", "").replace("cat__", "")
        if name.startswith("interest_"):
            label = "Student Interest"
        elif name.endswith("_failed") or name.endswith("_passed") or name in {"passed_count", "failed_count"}:
            label = "Passed/Failed Courses"
        elif name == "gpa":
            label = "GPA"
        elif name == "excellent_count":
            label = "Excellent Course Count"
        elif name == "weak_count":
            label = "Weak Course Count"
        else:
            label = name.replace("_grade", "").replace("_", " ").title() + " Grade"
        grouped[label] += float(importance)
    total = sum(grouped.values()) or 1.0
    return dict(sorted({k: v / total for k, v in grouped.items()}.items(), key=lambda x: x[1], reverse=True))


# This function trains the models, evaluates them, mines rules, and saves everything for the app.
# بالمصري: دي أهم دالة في التدريب، بتطلع الموديلات والنتائج اللي الموقع بيستخدمها.
def train_and_save() -> dict:
    print("=" * 64)
    print("  Smart Academic Advisor - Model Training")
    print("=" * 64)

    df = pd.read_csv("data/students.csv")
    print(f"\n[OK] Loaded dataset: {len(df)} students")

    X = df[FEATURES].copy()
    y = df[TARGET]

    # The target track names are encoded as numbers for scikit-learn models.
    # بالمصري: بنحوّل أسماء التراكات لأرقام أثناء التدريب ونرجعها أسماء وقت العرض.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Stratify keeps the same track distribution in both train and test sets.
    # بالمصري: عشان كل تراك يبقى ممثل بعدل في التدريب والاختبار.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # The project compares three classifiers required for the discussion.
    # Random Forest is the strongest default, Decision Tree is interpretable, and KNN supports similar-student evidence.
    # بالمصري: كل موديل ليه فايدة في الشرح، مش مجرد تكرار.
    model_configs = {
        "decision_tree": build_pipeline(DecisionTreeClassifier(max_depth=12, min_samples_split=8, random_state=42)),
        "random_forest": build_pipeline(RandomForestClassifier(n_estimators=260, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        "knn": build_pipeline(KNeighborsClassifier(n_neighbors=9, weights="distance", metric="euclidean")),
    }

    trained = {}
    for name, model in model_configs.items():
        # Train, test, cross-validate, and store the evaluation results for the dashboard.
        # بالمصري: هنا بنحسب accuracy وCV وconfusion matrix لكل موديل.
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv = cross_val_score(model, X, y_encoded, cv=5, scoring="accuracy")
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_))).tolist()
        trained[name] = {
            "model": model,
            "accuracy": round(acc * 100, 2),
            "cv_mean": round(float(cv.mean()) * 100, 2),
            "cv_std": round(float(cv.std()) * 100, 2),
            "report": report,
            "confusion_matrix": cm,
        }
        print(f"  {name.replace('_', ' ').title():<22} Acc: {acc * 100:6.2f}%   CV: {cv.mean() * 100:6.2f}% +/- {cv.std() * 100:4.2f}%")

    print("\n[...] Mining association rules...")
    # Apriori needs transaction-style data, so each student is converted into readable items first.
    # بالمصري: بنحوّل كل طالب لسلة items عشان نطلع منها rules.
    transactions = build_transactions(df)
    encoder = TransactionEncoder()
    encoded = encoder.fit_transform(transactions)
    transaction_df = pd.DataFrame(encoded, columns=encoder.columns_)
    freq = apriori(transaction_df, min_support=0.035, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.55)
    if rules.empty:
        rules_sorted = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    else:
        # Keep rules that include a track pattern so they can support advisor explanations.
        # بالمصري: مش عايزين أي rule وخلاص، عايزين rules مرتبطة بالتراكات.
        rules = rules[rules["consequents"].apply(lambda x: any(str(item).startswith("Track_") for item in x))]
        rules = rules.sort_values(["lift", "confidence"], ascending=False).head(80).copy()
        rules["antecedents"] = rules["antecedents"].apply(lambda x: sorted(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: sorted(list(x)))
        rules_sorted = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    print(f"[OK] Found {len(rules_sorted)} strong track association rules")

    os.makedirs("models", exist_ok=True)
    # The Flask app loads this one payload instead of retraining models every time it starts.
    # بالمصري: الموقع بيفتح الملف ده جاهز بدل ما يدرب من الأول كل مرة.
    payload = {
        "models": {name: data["model"] for name, data in trained.items()},
        "label_encoder": le,
        "accuracies": {name: data["accuracy"] for name, data in trained.items()},
        "cv_scores": {name: {"mean": data["cv_mean"], "std": data["cv_std"]} for name, data in trained.items()},
        "reports": {name: data["report"] for name, data in trained.items()},
        "confusion_matrices": {name: data["confusion_matrix"] for name, data in trained.items()},
        "feature_importance": cleaned_feature_importance(trained["random_forest"]["model"]),
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "grade_features": GRADE_FEATURES,
        "association_rules": rules_sorted,
        "classes": list(le.classes_),
        "training_features": X.reset_index(drop=True),
        "historical_students": df[["student_id", "gpa", "interest", "recommended_track", *GRADE_FEATURES, "failed_courses"]].reset_index(drop=True),
    }

    with open("models/trained_models.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("[OK] Models saved -> models/trained_models.pkl")
    print("\n" + "=" * 64)
    print("  Training complete. Run: python app.py")
    print("=" * 64)
    return payload


if __name__ == "__main__":
    train_and_save()
