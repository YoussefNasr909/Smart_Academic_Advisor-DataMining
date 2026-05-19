"""
generate_dirty_data.py
Generates a realistic MESSY / UNCLEAN synthetic dataset for the
Smart Academic Advisor preprocessing demonstration.

Issues intentionally injected:
  1.  Missing / null values           – GPA, grades, interest (~12-18 % per field)
  2.  Duplicate records               – ~5 % of rows duplicated
  3.  Inconsistent interest casing    – "ai/ml", "AI/ML", "Ai/Ml", "  AI/ML  "
  4.  Typos in categorical values     – "Neetworks", "Softwre Eng", "Dat Science"
  5.  Outlier / impossible grades     – -10, 150, 999, -999
  6.  Outlier / impossible GPA        – -1.5, 5.2, 9.99
  7.  Grade stored as string          – "85%", "90 pts", "seventy-two", "N/A"
  8.  GPA stored as string            – "3.45a", "N/A", "  2.7  "
  9.  Inconsistent student_id format  – "STU0001", "stu-0001", "STU 0001", "1"
  10. Whitespace padding in strings   – "  Networks  ", " Data Science "
  11. Special characters in values    – "AI/ML@2024", "Web#Dev"
  12. Binary flag inconsistencies     – passed + failed both = 1, or both = 0
  13. Numeric columns stored as mixed – some rows have float grades (85.7)
  14. Extra junk column               – "extra_info" with random garbage
  15. Inconsistent None encoding      – "none", "NONE", "N/A", "na", "-", ""

Run: python generate_dirty_data.py
Output: data/students_dirty.csv
"""
from __future__ import annotations

import os
import random
import string
from typing import Dict, List

import numpy as np
import pandas as pd

# ── Re-use the clean generation logic ─────────────────────────────────────────
# These course names are used to build grade, pass, and fail columns consistently.
# بالمصري: دي أسماء المواد الأساسية اللي المشروع كله ماشي عليها.
COURSE_MAP = {
    "math": "Mathematics",
    "programming": "Programming",
    "data_structures": "Data Structures",
    "algorithms": "Algorithms",
    "databases": "Databases",
    "networks": "Networks",
    "ai": "Artificial Intelligence",
    "web": "Web Development",
}

# Each track has a matching interest, important core courses, and realistic grade ranges.
# These rules make the synthetic data meaningful instead of completely random.
# بالمصري: هنا بنقول كل تراك محتاج مواد إيه ودرجاتها المفروض تبقى عاملة إزاي.
TRACKS = {
    "AI/Machine Learning": {
        "interest": "AI/ML",
        "core": ["math", "programming", "data_structures", "algorithms", "ai"],
        "ranges": {
            "math": (72, 100), "programming": (68, 100), "data_structures": (65, 98),
            "algorithms": (70, 100), "databases": (50, 84), "networks": (40, 76),
            "ai": (78, 100), "web": (40, 76),
        },
    },
    "Web Development": {
        "interest": "Web Development",
        "core": ["programming", "data_structures", "databases", "web"],
        "ranges": {
            "math": (50, 80), "programming": (72, 100), "data_structures": (62, 92),
            "algorithms": (50, 82), "databases": (68, 98), "networks": (50, 78),
            "ai": (40, 72), "web": (78, 100),
        },
    },
    "Network Engineering": {
        "interest": "Networks",
        "core": ["programming", "networks", "math"],
        "ranges": {
            "math": (60, 88), "programming": (56, 84), "data_structures": (52, 82),
            "algorithms": (54, 82), "databases": (48, 78), "networks": (78, 100),
            "ai": (40, 68), "web": (44, 72),
        },
    },
    "Data Science": {
        "interest": "Data Science",
        "core": ["math", "programming", "algorithms", "databases", "ai"],
        "ranges": {
            "math": (74, 100), "programming": (66, 96), "data_structures": (60, 92),
            "algorithms": (66, 94), "databases": (72, 100), "networks": (40, 72),
            "ai": (66, 98), "web": (45, 74),
        },
    },
    "Software Engineering": {
        "interest": "Software Engineering",
        "core": ["programming", "data_structures", "algorithms", "databases", "web"],
        "ranges": {
            "math": (55, 84), "programming": (74, 100), "data_structures": (72, 100),
            "algorithms": (66, 96), "databases": (66, 92), "networks": (50, 78),
            "ai": (45, 74), "web": (64, 90),
        },
    },
}

# These mappings connect student interests to the academic tracks used in the dataset.
# بالمصري: دي بتربط اختيار الطالب للتخصص بالتراك المناسب في الداتا.
INTEREST_TO_TRACK = {v["interest"]: k for k, v in TRACKS.items()}
INTERESTS = list(INTEREST_TO_TRACK.keys()) + ["Undecided"]


# This function keeps generated grades inside the valid 0 to 100 range.
# بالمصري: لو رقم طلع بره الرينج، بنرجعه بين 0 و100.
def clipped_grade(value: float) -> int:
    return int(np.clip(round(value), 0, 100))


# This function estimates GPA from course grades so each student has an academic summary.
# بالمصري: بنحوّل متوسط الدرجات لـ GPA من 4 عشان يبقى عندنا ملخص لمستوى الطالب.
def calculate_gpa(grades: Dict[str, int]) -> float:
    avg = float(np.mean(list(grades.values())))
    return round(float(np.clip(avg / 25 + np.random.normal(0, 0.04), 0, 4)), 2)


# This function scores one track using grades, GPA, interest, and weak/failed courses.
# بالمصري: هنا بنحسب التراك ده مناسب للطالب قد إيه قبل ما نختار أحسن تراك.
def score_track(track: str, grades: Dict[str, int], interest: str, gpa: float) -> float:
    profile = TRACKS[track]
    core_avg = np.mean([grades[c] for c in profile["core"]])
    all_avg = np.mean(list(grades.values()))
    failed_core = sum(1 for c in profile["core"] if grades[c] < 60)
    weak_core = sum(1 for c in profile["core"] if grades[c] < 70)
    score = (0.58 * core_avg) + (0.17 * all_avg) + (0.25 * gpa * 25)
    if interest == profile["interest"]:
        score += 13
    elif interest == "Undecided":
        score += 1
    score -= failed_core * 18
    score -= weak_core * 4
    return float(score)


# This function chooses the best academic track by comparing all track scores.
# بالمصري: بنجرّب كل التراكات ونختار أعلى score.
def recommend_track(grades: Dict[str, int], interest: str, gpa: float) -> str:
    scores = {track: score_track(track, grades, interest, gpa) for track in TRACKS}
    return max(scores, key=scores.get)


# This function creates the clean synthetic student dataset before adding data problems.
# بالمصري: دي بتعمل داتا طلاب نظيفة الأول، وبعدها بنبوّظها عمدا للتدريب على التنضيف.
def generate_clean_records(n: int = 900, seed: int = 42) -> pd.DataFrame:
    """Generate the base clean records (identical logic to generate_data.py)."""
    # Fixed seeds make the generated dataset reproducible for training and discussion.
    np.random.seed(seed)
    random.seed(seed)
    track_names = list(TRACKS.keys())
    records: List[Dict[str, object]] = []

    for i in range(n):
        # A base track is chosen first, then grades are generated around that track's profile.
        # بالمصري: بنختار تراك مبدئي للطالب عشان درجاته تطلع منطقية مش عشوائية تماما.
        base_track = np.random.choice(track_names)
        profile = TRACKS[base_track]

        grades = {}
        for course_key in COURSE_MAP:
            # Normal distribution makes most grades close to the expected range midpoint.
            # بالمصري: أغلب الدرجات بتطلع حوالين المتوسط، زي الدرجات الحقيقية غالبا.
            low, high = profile["ranges"][course_key]
            grade = np.random.normal((low + high) / 2, max(6, (high - low) / 5))
            grades[course_key] = clipped_grade(grade)

        # These random changes stop the synthetic data from looking too perfect.
        # بالمصري: بنضيف طالب ضعيف في مادة أو قوي في مادة عشان الداتا تبقى واقعية.
        if np.random.rand() < 0.18:
            weak_course = np.random.choice(list(COURSE_MAP.keys()))
            grades[weak_course] = int(np.random.randint(35, 60))
        if np.random.rand() < 0.22:
            boost_course = np.random.choice(list(COURSE_MAP.keys()))
            grades[boost_course] = int(np.random.randint(86, 101))

        interest_roll = np.random.rand()
        if interest_roll < 0.72:
            interest = profile["interest"]
        elif interest_roll < 0.92:
            interest = np.random.choice(
                [x for x in INTERESTS if x not in {profile["interest"], "Undecided"}]
            )
        else:
            interest = "Undecided"

        gpa = calculate_gpa(grades)
        passed = [COURSE_MAP[k] for k in COURSE_MAP if grades[k] >= 60]
        failed = [COURSE_MAP[k] for k in COURSE_MAP if grades[k] < 60]
        recommended = recommend_track(grades, interest, gpa)

        row: Dict[str, object] = {
            "student_id": f"STU{i + 1:04d}",
            "gpa": gpa,
            "interest": interest,
            "passed_courses": "|".join(passed),
            "failed_courses": "|".join(failed) if failed else "None",
            "passed_count": len(passed),
            "failed_count": len(failed),
            "excellent_count": sum(1 for v in grades.values() if v >= 85),
            "weak_count": sum(1 for v in grades.values() if v < 70),
            "recommended_track": recommended,
        }
        for key, value in grades.items():
            row[f"{key}_grade"] = value
            row[f"{key}_passed"] = int(value >= 60)
            row[f"{key}_failed"] = int(value < 60)
        records.append(row)

    ordered_grade_cols = [f"{k}_grade" for k in COURSE_MAP]
    ordered_binary_cols: List[str] = []
    for k in COURSE_MAP:
        ordered_binary_cols.extend([f"{k}_passed", f"{k}_failed"])
    ordered_cols = [
        "student_id", "gpa", *ordered_grade_cols, "interest",
        "passed_courses", "failed_courses", "passed_count", "failed_count",
        "excellent_count", "weak_count", *ordered_binary_cols, "recommended_track",
    ]
    return pd.DataFrame(records)[ordered_cols]


# ── Noise injection helpers ────────────────────────────────────────────────────

# These lists contain the messy values injected into the clean dataset.
# They help demonstrate realistic preprocessing problems in the project discussion.
# بالمصري: دي أمثلة الأخطاء اللي بندخلها عمدا عشان نثبت إن preprocessing شغال.
INTEREST_TYPOS = {
    "AI/ML":                ["ai/ml", "AI/ML", "Ai/Ml", "ai ml", "A.I/ML", "AI ML", "Ai/ml"],
    "Web Development":      ["web development", "Web Dev", "WEB DEVELOPMENT", "web developement",
                             "Web Develoment", "WebDevelopment", "web  development"],
    "Networks":             ["networks", "NETWORKS", "Neetworks", "Netwroks", "network",
                             "Networks ", " Networks", "Netowrks"],
    "Data Science":         ["data science", "DATA SCIENCE", "Dat Science", "Data Sceince",
                             "DataScience", " Data Science ", "data  science"],
    "Software Engineering": ["software engineering", "SOFTWARE ENGINEERING", "Softwre Engineering",
                             "Software Eng", "SoftwareEngineering", " software engineering ",
                             "Sofware Engineering"],
    "Undecided":            ["undecided", "UNDECIDED", "Undecideed", "un decided", "undecidied"],
}

NONE_ENCODINGS = ["none", "NONE", "N/A", "na", "NA", "-", "", "null", "NULL", "nan"]

GRADE_STRING_SUFFIXES = ["%", " pts", " points", " marks", "/100"]

WORD_GRADES = {
    0: "zero", 10: "ten", 20: "twenty", 30: "thirty", 40: "forty",
    50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty",
    90: "ninety", 100: "one hundred",
}

OUTLIER_GRADES   = [-999, -10, -5, 101, 110, 150, 200, 999]
OUTLIER_GPAS     = [-1.5, -0.5, 4.1, 4.5, 5.0, 5.2, 9.99]


# This function makes student IDs messy to simulate real data entry mistakes.
# بالمصري: بنغير شكل الـ ID كأنه اتكتب غلط في السيستم.
def corrupt_student_id(sid: str, rng: np.random.Generator) -> str:
    """Return the student ID in one of several messy formats."""
    num = sid[3:]  # e.g. "0042"
    choice = rng.integers(0, 6)
    if choice == 0:
        return sid.lower()                        # "stu0042"
    if choice == 1:
        return f"STU-{num}"                       # "STU-0042"
    if choice == 2:
        return f"STU {num}"                       # "STU 0042"
    if choice == 3:
        return str(int(num))                      # "42"
    if choice == 4:
        return f"  {sid}  "                       # "  STU0042  "
    return f"{sid}#"                              # "STU0042#"


# This function changes interest values into typos or inconsistent formats.
# بالمصري: بنعمل typos في الاهتمامات زي اللي بتحصل في إدخال البيانات.
def corrupt_interest(interest: str, rng: np.random.Generator) -> str:
    variants = INTEREST_TYPOS.get(interest, [interest])
    return rng.choice(variants)


# This function changes a numeric grade into a messy string value.
# بالمصري: بنخلي الدرجة تبقى نص زي "85%" أو "90 pts" عشان preprocessing ينضفها.
def corrupt_grade_as_string(grade: int, rng: np.random.Generator) -> str:
    choice = rng.integers(0, 4)
    if choice == 0:
        suffix = rng.choice(GRADE_STRING_SUFFIXES)
        return f"{grade}{suffix}"
    if choice == 1:
        # Find nearest word-grade
        nearest = min(WORD_GRADES.keys(), key=lambda k: abs(k - grade))
        return WORD_GRADES[nearest]
    if choice == 2:
        return "N/A"
    return f"  {grade}  "   # whitespace padding as string


# This function changes a numeric GPA into a messy string value.
# بالمصري: بنخلي الـ GPA يتكتب بشكل ملخبط زي "3.45a".
def corrupt_gpa_as_string(gpa: float, rng: np.random.Generator) -> str:
    choice = rng.integers(0, 4)
    if choice == 0:
        return f"{gpa}a"        # trailing letter
    if choice == 1:
        return "N/A"
    if choice == 2:
        return f"  {gpa}  "     # whitespace padding
    return str(gpa) + "!"       # special character


# This function adds missing values, duplicates, outliers, and typos for preprocessing practice.
# بالمصري: دي أهم دالة بتبوّظ الداتا عشان نوري شغل الـ data cleaning.
def inject_noise(df: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    """Apply all data quality issues to the clean DataFrame in-place copy."""
    rng = np.random.default_rng(seed)
    # Convert all columns to object dtype so we can freely mix strings and numbers
    df = df.copy().astype(object)
    n = len(df)
    grade_cols = [f"{k}_grade" for k in COURSE_MAP]

    # ── 1. Add an extra junk column ───────────────────────────────────────────
    junk = []
    for _ in range(n):
        choice = rng.integers(0, 4)
        if choice == 0:
            junk.append("".join(rng.choice(list(string.ascii_letters), size=int(rng.integers(3, 8)))))
        elif choice == 1:
            junk.append(rng.integers(-100, 500))
        elif choice == 2:
            junk.append(None)
        else:
            junk.append(f"INFO_{rng.integers(1, 9999)}")
    # This irrelevant column will be removed later during preprocessing.
    # بالمصري: عمود مالوش لازمة عشان نوري إننا بنشيل noise.
    df["extra_info"] = junk

    # ── 2. Corrupt student IDs (~20 % of rows) ────────────────────────────────
    id_corrupt_idx = rng.choice(n, size=int(n * 0.20), replace=False)
    for i in id_corrupt_idx:
        df.at[i, "student_id"] = corrupt_student_id(str(df.at[i, "student_id"]), rng)

    # ── 3. Corrupt interest values (~40 % of rows get typos/case issues) ──────
    interest_corrupt_idx = rng.choice(n, size=int(n * 0.40), replace=False)
    for i in interest_corrupt_idx:
        df.at[i, "interest"] = corrupt_interest(str(df.at[i, "interest"]), rng)

    # ── 4. Missing values in GPA (~10 % of rows) ──────────────────────────────
    gpa_null_idx = rng.choice(n, size=int(n * 0.10), replace=False)
    df.loc[gpa_null_idx, "gpa"] = np.nan

    # ── 5. Outlier GPA values (~3 % of rows) ─────────────────────────────────
    gpa_outlier_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    for i in gpa_outlier_idx:
        df.at[i, "gpa"] = float(rng.choice(OUTLIER_GPAS))

    # ── 6. GPA stored as malformed string (~4 % of rows) ─────────────────────
    gpa_str_idx = rng.choice(
        [i for i in range(n) if i not in set(gpa_null_idx) and i not in set(gpa_outlier_idx)],
        size=int(n * 0.04), replace=False,
    )
    for i in gpa_str_idx:
        df.at[i, "gpa"] = corrupt_gpa_as_string(float(df.at[i, "gpa"]), rng)

    # ── 7. Missing grades (~8 % per grade column) ─────────────────────────────
    for col in grade_cols:
        null_idx = rng.choice(n, size=int(n * 0.08), replace=False)
        df.loc[null_idx, col] = np.nan

    # ── 8. Outlier grades (~2 % per grade column) ────────────────────────────
    for col in grade_cols:
        out_idx = rng.choice(n, size=int(n * 0.02), replace=False)
        for i in out_idx:
            df.at[i, col] = float(rng.choice(OUTLIER_GRADES))

    # ── 9. Grade as malformed string (~3 % per grade column) ─────────────────
    for col in grade_cols:
        str_idx = rng.choice(n, size=int(n * 0.03), replace=False)
        for i in str_idx:
            raw = df.at[i, col]
            try:
                grade_val = int(float(raw)) if not pd.isna(raw) else 70
            except (TypeError, ValueError):
                grade_val = 70
            df.at[i, col] = corrupt_grade_as_string(grade_val, rng)

    # ── 10. Whitespace / special char in interest (~5 % extra) ───────────────
    ws_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    for i in ws_idx:
        raw = str(df.at[i, "interest"])
        choice = rng.integers(0, 3)
        if choice == 0:
            df.at[i, "interest"] = f"  {raw}  "
        elif choice == 1:
            df.at[i, "interest"] = raw + "@2024"
        else:
            df.at[i, "interest"] = raw + "#"

    # ── 11. Inconsistent None encoding in failed_courses ─────────────────────
    none_idx = rng.choice(n, size=int(n * 0.15), replace=False)
    for i in none_idx:
        if str(df.at[i, "failed_courses"]).strip().lower() in {"none", ""}:
            df.at[i, "failed_courses"] = rng.choice(NONE_ENCODINGS)

    # ── 12. Binary flag inconsistencies (~3 % of rows per course) ────────────
    for course in COURSE_MAP:
        bad_idx = rng.choice(n, size=int(n * 0.03), replace=False)
        for i in bad_idx:
            flip = rng.integers(0, 3)
            if flip == 0:
                df.at[i, f"{course}_passed"] = 1  # both 1
                df.at[i, f"{course}_failed"] = 1
            elif flip == 1:
                df.at[i, f"{course}_passed"] = 0  # both 0
                df.at[i, f"{course}_failed"] = 0
            else:
                df.at[i, f"{course}_passed"] = int(rng.integers(0, 2))  # random mismatch

    # ── 13. passed_count / failed_count inconsistencies ──────────────────────
    count_bad_idx = rng.choice(n, size=int(n * 0.06), replace=False)
    for i in count_bad_idx:
        df.at[i, "passed_count"] = int(rng.integers(0, 12))
        df.at[i, "failed_count"] = int(rng.integers(0, 9))

    # ── 14. Float grades where integers are expected ──────────────────────────
    float_grade_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    col_choice = rng.choice(grade_cols, size=len(float_grade_idx))
    for i, col in zip(float_grade_idx, col_choice):
        raw = df.at[i, col]
        try:
            base = float(raw)
            df.at[i, col] = round(base + rng.uniform(0.1, 0.9), 1)
        except (TypeError, ValueError):
            pass

    # ── 15. Add duplicate rows (~5 % of dataset) ─────────────────────────────
    dup_count = int(n * 0.05)
    dup_src_idx = rng.choice(n, size=dup_count, replace=False)
    dup_rows = df.iloc[dup_src_idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    # ── 16. Shuffle the whole dataset ────────────────────────────────────────
    df = df.sample(frac=1, random_state=int(seed)).reset_index(drop=True)

    return df


# This function runs the full dirty-data generation step and saves the raw dataset.
# بالمصري: دي نقطة تشغيل ملف توليد الداتا قبل التنضيف.
def generate_dirty_data(n: int = 900, seed: int = 42) -> pd.DataFrame:
    print("=" * 64)
    print("  Smart Academic Advisor - Dirty Dataset Generator")
    print("=" * 64)

    print("\n[1/3] Generating clean base records...")
    clean_df = generate_clean_records(n=n, seed=seed)
    print(f"       Base records: {len(clean_df)}")

    print("\n[2/3] Injecting data quality issues...")
    dirty_df = inject_noise(clean_df, seed=seed + 10)

    issues_summary = {
        "Total rows (incl. duplicates)": len(dirty_df),
        "Approximate duplicates injected": int(n * 0.05),
        "Columns with missing values": dirty_df.isnull().any().sum(),
        "Total missing cells": int(dirty_df.isnull().sum().sum()),
        "Rows with >= 1 missing value": int(dirty_df.isnull().any(axis=1).sum()),
    }
    for k, v in issues_summary.items():
        print(f"       {k}: {v}")

    print("\n[3/3] Saving dirty dataset...")
    os.makedirs("data", exist_ok=True)
    dirty_df.to_csv("data/students_dirty.csv", index=False)
    print(f"\n[OK] Dirty dataset saved -> data/students_dirty.csv")
    print(f"     Rows: {len(dirty_df)}  |  Columns: {len(dirty_df.columns)}")
    print("\nIssues injected per category:")
    print("  [X] Missing values          (GPA ~10%, grades ~8% each, interest NaN)")
    print("  [X] Duplicate rows          (~5% of dataset)")
    print("  [X] Interest typos/case     (~40% of interest values)")
    print("  [X] Outlier grades          (<0 or >100, e.g. -999, 150)")
    print("  [X] Outlier GPA             (<0 or >4, e.g. -1.5, 9.99)")
    print("  [X] Grade as string         ('85%', 'seventy', 'N/A')")
    print("  [X] GPA as string           ('3.45a', '  2.7  ', 'N/A')")
    print("  [X] ID format inconsistency (stu-0001, STU 0001, 1)")
    print("  [X] Whitespace/special char (in interest, student_id)")
    print("  [X] None encoding variants  ('none','N/A','na','-','')")
    print("  [X] Binary flag corruption  (passed+failed both 0 or both 1)")
    print("  [X] Count inconsistencies   (passed_count, failed_count)")
    print("  [X] Float where int expected(grades as 85.7)")
    print("  [X] Junk column             (extra_info with random noise)")
    print("=" * 64)
    return dirty_df


if __name__ == "__main__":
    generate_dirty_data()
