"""
preprocess_data.py
Comprehensive preprocessing and cleaning pipeline for the Smart Academic Advisor.

Pipeline steps
--------------
  Step 1   Load dirty dataset
  Step 2   Drop junk / irrelevant columns
  Step 3   Standardise column names
  Step 4   Detect and remove duplicate rows
  Step 5   Clean and standardise student_id
  Step 6   Parse and clean GPA (string -> float, clamp 0-4, fill missing)
  Step 7   Parse and clean grade columns (string -> int, clamp 0-100, fill missing)
  Step 8   Standardise interest categorical values
  Step 9   Recompute binary pass/fail flags from cleaned grades
  Step 10  Recompute summary counts from cleaned grades
  Step 11  Standardise failed_courses 'None' encoding
  Step 12  Validate and report final dataset quality
  Step 13  Save clean dataset + human-readable preprocessing report

Run: python preprocess_data.py
Input:  data/students_dirty.csv
Output: data/students.csv
        data/preprocessing_report.txt
"""
from __future__ import annotations

import os
import re
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -- Constants -----------------------------------------------------------------

COURSE_KEYS = ["math", "programming", "data_structures", "algorithms",
               "databases", "networks", "ai", "web"]
GRADE_COLS  = [f"{k}_grade" for k in COURSE_KEYS]
PASSED_COLS = [f"{k}_passed" for k in COURSE_KEYS]
FAILED_COLS = [f"{k}_failed" for k in COURSE_KEYS]

COURSE_LABELS = {
    "math_grade":            "Mathematics",
    "programming_grade":     "Programming",
    "data_structures_grade": "Data Structures",
    "algorithms_grade":      "Algorithms",
    "databases_grade":       "Databases",
    "networks_grade":        "Networks",
    "ai_grade":              "Artificial Intelligence",
    "web_grade":             "Web Development",
}

# Canonical interest values + every known messy variant
INTEREST_CANONICAL: Dict[str, str] = {
    # AI/ML variants
    "ai/ml":                          "AI/ML",
    "aiml":                           "AI/ML",
    "ai ml":                          "AI/ML",
    "a.i/ml":                         "AI/ML",
    "a.i. / ml":                      "AI/ML",
    "artificialintelligence/machinelearning": "AI/ML",
    "ai/machinelearning":             "AI/ML",
    # Web Development variants
    "web development":                "Web Development",
    "web dev":                        "Web Development",
    "webdevelopment":                 "Web Development",
    "web developement":               "Web Development",
    "web develoment":                 "Web Development",
    "web  development":               "Web Development",
    # Networks variants
    "networks":                       "Networks",
    "network":                        "Networks",
    "neetworks":                      "Networks",
    "netwroks":                       "Networks",
    "netowrks":                       "Networks",
    # Data Science variants
    "data science":                   "Data Science",
    "datascience":                    "Data Science",
    "dat science":                    "Data Science",
    "data sceince":                   "Data Science",
    "data  science":                  "Data Science",
    # Software Engineering variants
    "software engineering":           "Software Engineering",
    "softwareengineering":            "Software Engineering",
    "softwre engineering":            "Software Engineering",
    "sofware engineering":            "Software Engineering",
    "software eng":                   "Software Engineering",
    # Undecided variants
    "undecided":                      "Undecided",
    "undecideed":                     "Undecided",
    "un decided":                     "Undecided",
    "undecidied":                     "Undecided",
}

VALID_INTERESTS = {
    "AI/ML", "Web Development", "Networks",
    "Data Science", "Software Engineering", "Undecided",
}

NONE_VARIANTS = {"none", "n/a", "na", "-", "", "null", "nan", "nil"}

WORD_TO_NUMBER = {
    "zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "one hundred": 100, "hundred": 100,
}


# -- Logging helpers -----------------------------------------------------------

class PreprocessingLog:
    """Collects all cleaning actions for the final report."""

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def log(self, step: int, title: str, action: str,
            rows_before: int, rows_after: int,
            cells_fixed: int = 0, notes: str = "") -> None:
        self.steps.append({
            "step": step,
            "title": title,
            "action": action,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_affected": rows_before - rows_after if rows_before != rows_after else cells_fixed,
            "notes": notes,
        })

    def summary_table(self) -> str:
        lines = []
        lines.append(f"{'Step':<5} {'Title':<38} {'Rows Before':>11} {'Rows After':>10} {'Changed':>9}")
        lines.append("-" * 78)
        for s in self.steps:
            lines.append(
                f"{s['step']:<5} {s['title']:<38} {s['rows_before']:>11,} "
                f"{s['rows_after']:>10,} {s['rows_affected']:>9,}"
            )
        return "\n".join(lines)

    def detail_lines(self) -> str:
        lines = []
        for s in self.steps:
            lines.append(f"\n-- Step {s['step']}: {s['title']} {'-' * max(0, 60 - len(s['title']))}")
            lines.append(f"   Action  : {s['action']}")
            lines.append(f"   Before  : {s['rows_before']:,} rows")
            lines.append(f"   After   : {s['rows_after']:,} rows")
            if s['rows_affected']:
                lines.append(f"   Changed : {s['rows_affected']:,}")
            if s['notes']:
                for note_line in textwrap.wrap(s['notes'], width=72):
                    lines.append(f"   Note    : {note_line}")
        return "\n".join(lines)


# -- Value parsers -------------------------------------------------------------

def parse_grade(raw: Any) -> Optional[float]:
    """
    Convert a raw grade cell to a float (or None if unparseable).
    Handles: int, float, "85%", "90 pts", "seventy", "  85  ", NaN.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip().lower()
    if not text or text in NONE_VARIANTS:
        return None
    # Remove common suffixes
    text = re.sub(r'\s*(pts?|points?|marks?|/100|%)\s*$', '', text).strip()
    # Word-number lookup
    for word, num in WORD_TO_NUMBER.items():
        if text == word:
            return float(num)
    # Numeric (int or float)
    try:
        return float(text)
    except ValueError:
        # Try extracting leading numeric part
        m = re.match(r'^(-?\d+(?:\.\d+)?)', text)
        if m:
            return float(m.group(1))
        return None


def parse_gpa(raw: Any) -> Optional[float]:
    """Convert a raw GPA cell to float, or None if unparseable."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip().lower()
    if not text or text in NONE_VARIANTS:
        return None
    # Remove trailing non-numeric characters
    cleaned = re.sub(r'[^0-9.]', '', text)
    try:
        return float(cleaned)
    except ValueError:
        return None


def standardise_interest(raw: Any) -> str:
    """Map a messy interest string to one of the six canonical values."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "Undecided"
    # Strip whitespace and special characters
    text = re.sub(r'[^a-zA-Z0-9/ ]', '', str(raw)).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    if text in INTEREST_CANONICAL:
        return INTEREST_CANONICAL[text]
    # Case-insensitive prefix/substring match
    for variant, canonical in INTEREST_CANONICAL.items():
        if text.startswith(variant) or variant.startswith(text[:6]):
            return canonical
    return "Undecided"


def clean_student_id(raw: Any) -> str:
    """Normalise student_id to STU####."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "UNKNOWN"
    text = re.sub(r'[^a-zA-Z0-9]', '', str(raw)).strip().upper()
    # Extract the numeric part
    m = re.search(r'(\d+)', text)
    if m:
        num = int(m.group(1))
        return f"STU{num:04d}"
    return text if text else "UNKNOWN"


# -- Main pipeline -------------------------------------------------------------

def run_pipeline(input_path: str = "data/students_dirty.csv",
                 output_path: str = "data/students.csv",
                 report_path: str = "data/preprocessing_report.txt") -> pd.DataFrame:

    log = PreprocessingLog()
    print("=" * 64)
    print("  Smart Academic Advisor - Preprocessing Pipeline")
    print("=" * 64)

    # -- Step 1: Load ----------------------------------------------------------
    print("\n[Step 1] Loading dirty dataset...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Dirty dataset not found at '{input_path}'.\n"
            "Run:  python generate_dirty_data.py"
        )
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    # Replace literal "nan" strings produced by pandas with actual NaN
    df.replace({"nan": np.nan, "NaN": np.nan, "NAN": np.nan}, inplace=True)
    initial_rows = len(df)
    initial_cols = len(df.columns)
    print(f"       Loaded {initial_rows:,} rows × {initial_cols} columns")
    print(f"       Missing cells: {df.isnull().sum().sum():,}")
    log.log(1, "Load dirty dataset", f"Read '{input_path}'",
            initial_rows, initial_rows,
            notes=f"{initial_cols} columns, {df.isnull().sum().sum()} missing cells total")

    # -- Step 2: Drop junk columns ---------------------------------------------
    print("\n[Step 2] Dropping irrelevant/junk columns...")
    expected_cols = (
        ["student_id", "gpa"] + GRADE_COLS +
        ["interest", "passed_courses", "failed_courses",
         "passed_count", "failed_count", "excellent_count", "weak_count"] +
        PASSED_COLS + FAILED_COLS + ["recommended_track"]
    )
    junk_cols = [c for c in df.columns if c not in expected_cols]
    if junk_cols:
        df.drop(columns=junk_cols, inplace=True)
        print(f"       Removed columns: {junk_cols}")
    else:
        print("       No junk columns found.")
    log.log(2, "Drop junk columns",
            f"Removed: {junk_cols if junk_cols else 'none'}",
            initial_rows, len(df),
            notes=f"Kept {len(df.columns)} expected columns")

    # -- Step 3: Standardise column names -------------------------------------
    print("\n[Step 3] Standardising column names...")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.log(3, "Standardise column names", "Strip, lowercase, underscore",
            len(df), len(df), notes="All column names normalised")

    # -- Step 4: Remove duplicate rows ----------------------------------------
    print("\n[Step 4] Removing duplicate rows...")
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    after = len(df)
    removed = before - after
    print(f"       Duplicates removed: {removed:,}  ({removed / before * 100:.1f}%)")
    log.log(4, "Remove duplicates",
            "drop_duplicates() on all columns",
            before, after,
            notes=f"{removed} exact duplicate rows dropped")

    # -- Step 5: Clean student_id ----------------------------------------------
    print("\n[Step 5] Cleaning student_id...")
    before_unique = df["student_id"].nunique()
    df["student_id"] = df["student_id"].apply(clean_student_id)
    # After ID normalisation, drop any new duplicates (e.g. "STU 0042" == "STU0042")
    before2 = len(df)
    df.drop_duplicates(subset="student_id", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    id_dups_removed = before2 - len(df)
    print(f"       Unique IDs before: {before_unique} -> after: {df['student_id'].nunique()}")
    if id_dups_removed:
        print(f"       ID-collision duplicates removed: {id_dups_removed}")
    log.log(5, "Clean student_id",
            "Normalise format to STU#### then drop ID duplicates",
            before2, len(df),
            notes=f"Patterns handled: stu-####, STU ####, bare number, trailing # etc.")

    n = len(df)  # working row count after deduplication

    # -- Step 6: Parse and clean GPA -------------------------------------------
    print("\n[Step 6] Cleaning GPA...")
    gpa_raw = df["gpa"].copy()
    df["gpa"] = df["gpa"].apply(parse_gpa)
    df["gpa"] = pd.to_numeric(df["gpa"], errors="coerce")

    # Identify and report outliers before clamping
    outlier_gpa_mask = df["gpa"].notna() & ((df["gpa"] < 0) | (df["gpa"] > 4.0))
    n_gpa_outliers = int(outlier_gpa_mask.sum())
    df.loc[outlier_gpa_mask, "gpa"] = np.nan   # treat impossible values as missing

    n_gpa_missing = int(df["gpa"].isna().sum())
    # Fill missing GPA with median of valid values
    gpa_median = round(float(df["gpa"].median()), 2)
    df["gpa"] = df["gpa"].fillna(gpa_median)
    df["gpa"] = df["gpa"].clip(lower=0.0, upper=4.0).round(2)

    print(f"       Outlier GPA values nullified : {n_gpa_outliers}")
    print(f"       Missing GPA filled w/ median : {n_gpa_missing}  (median={gpa_median})")
    log.log(6, "Parse & clean GPA",
            "parse_gpa() -> numeric -> clamp [0,4] -> fill missing with median",
            n, n, cells_fixed=n_gpa_missing + n_gpa_outliers,
            notes=(f"String formats handled: '3.45a', '  2.7  ', 'N/A'.  "
                   f"{n_gpa_outliers} outliers (e.g. -1.5, 9.99) -> NaN then filled."))

    # -- Step 7: Parse and clean grade columns ---------------------------------
    print("\n[Step 7] Cleaning grade columns...")
    total_missing_grades = 0
    total_outlier_grades = 0
    total_str_grades     = 0

    for col in GRADE_COLS:
        # Detect how many were non-numeric strings before conversion
        non_numeric_before = df[col].apply(
            lambda x: not str(x).replace(".", "").replace("-", "").isdigit()
        ).sum()
        total_str_grades += int(non_numeric_before)

        df[col] = df[col].apply(parse_grade)
        df[col] = pd.to_numeric(df[col], errors="coerce")

        outlier_mask = df[col].notna() & ((df[col] < 0) | (df[col] > 100))
        n_outliers = int(outlier_mask.sum())
        total_outlier_grades += n_outliers
        df.loc[outlier_mask, col] = np.nan

        n_missing = int(df[col].isna().sum())
        total_missing_grades += n_missing

        col_median = round(float(df[col].median()), 0)
        df[col] = df[col].fillna(col_median)
        # Round float grades to int
        df[col] = df[col].fillna(col_median).clip(lower=0, upper=100).round(0).astype(int)

    print(f"       Non-numeric string grades resolved : {total_str_grades}")
    print(f"       Outlier grades (outside 0-100)     : {total_outlier_grades}")
    print(f"       Missing grade cells filled          : {total_missing_grades}")
    log.log(7, "Parse & clean grade columns",
            "parse_grade() -> numeric -> clamp [0,100] -> fill missing with column median",
            n, n, cells_fixed=total_missing_grades + total_outlier_grades + total_str_grades,
            notes=(f"String formats: '85%', '90 pts', 'seventy', 'N/A'.  "
                   f"{total_outlier_grades} impossible values (e.g. -999, 150) -> NaN then median-filled."))

    # -- Step 8: Standardise interest ------------------------------------------
    print("\n[Step 8] Standardising interest values...")
    before_dist = df["interest"].value_counts().to_dict()
    df["interest"] = df["interest"].apply(standardise_interest)
    after_dist  = df["interest"].value_counts().to_dict()
    non_canonical_before = sum(v for k, v in before_dist.items() if k not in VALID_INTERESTS)
    print(f"       Non-canonical values fixed: {non_canonical_before}")
    print(f"       Valid distribution after  : {dict(sorted(after_dist.items()))}")
    log.log(8, "Standardise interest",
            "Normalise case, strip whitespace/symbols, map typos to canonical value",
            n, n, cells_fixed=non_canonical_before,
            notes=("Handled: ai/ml, Ai/Ml, AI ML, Neetworks, Softwre Engineering, "
                   "Dat Science, trailing @/# symbols, excess whitespace."))

    # -- Step 9: Recompute binary pass/fail flags ------------------------------
    print("\n[Step 9] Recomputing binary pass/fail flags...")
    flags_fixed = 0
    for key in COURSE_KEYS:
        grade_col  = f"{key}_grade"
        passed_col = f"{key}_passed"
        failed_col = f"{key}_failed"
        correct_passed = (df[grade_col] >= 60).astype(int)
        correct_failed = (df[grade_col] < 60).astype(int)
        flags_fixed += int((df[passed_col].astype(str).str.strip() != correct_passed.astype(str)).sum())
        df[passed_col] = correct_passed
        df[failed_col] = correct_failed
    print(f"       Binary flag cells corrected: {flags_fixed}")
    log.log(9, "Recompute binary flags",
            "Derive {course}_passed / {course}_failed from cleaned grades (threshold 60)",
            n, n, cells_fixed=flags_fixed,
            notes="Eliminates both-1 / both-0 / mismatch corruptions from injection.")

    # -- Step 10: Recompute summary counts ------------------------------------
    print("\n[Step 10] Recomputing summary count columns...")
    df["passed_count"]   = df[PASSED_COLS].sum(axis=1).astype(int)
    df["failed_count"]   = df[FAILED_COLS].sum(axis=1).astype(int)
    df["excellent_count"] = (df[GRADE_COLS] >= 85).sum(axis=1).astype(int)
    df["weak_count"]      = (df[GRADE_COLS] < 70).sum(axis=1).astype(int)
    log.log(10, "Recompute summary counts",
            "passed_count, failed_count, excellent_count, weak_count from cleaned grades",
            n, n, notes="All four count columns recomputed deterministically.")

    # -- Step 11: Standardise failed_courses 'None' encoding ------------------
    print("\n[Step 11] Standardising failed_courses 'None' encoding...")
    none_variants_mask = df["failed_courses"].str.strip().str.lower().isin(NONE_VARIANTS)
    n_none_fixed = int(none_variants_mask.sum())
    df.loc[none_variants_mask, "failed_courses"] = "None"
    # Cross-validate: students with 0 failed grades should show "None"
    should_be_none = df["failed_count"] == 0
    df.loc[should_be_none, "failed_courses"] = "None"
    print(f"       None-variant cells normalised: {n_none_fixed}")
    log.log(11, "Standardise None encoding",
            "Map '', 'n/a', 'na', '-', 'null', 'none', 'nan' -> 'None' in failed_courses",
            n, n, cells_fixed=n_none_fixed,
            notes="Also cross-validated against failed_count to enforce consistency.")

    # -- Step 12: Final column ordering and type enforcement ------------------
    print("\n[Step 12] Enforcing column order and data types...")
    ordered_cols = (
        ["student_id", "gpa"] + GRADE_COLS +
        ["interest", "passed_courses", "failed_courses",
         "passed_count", "failed_count", "excellent_count", "weak_count"] +
        PASSED_COLS + FAILED_COLS + ["recommended_track"]
    )
    # Keep only expected columns (junk already removed in step 2)
    df = df[[c for c in ordered_cols if c in df.columns]]
    # Enforce types
    df["gpa"] = df["gpa"].astype(float)
    for col in GRADE_COLS + PASSED_COLS + FAILED_COLS:
        df[col] = df[col].astype(int)
    for col in ["passed_count", "failed_count", "excellent_count", "weak_count"]:
        df[col] = df[col].astype(int)
    log.log(12, "Enforce types & column order",
            "Cast GPA->float, grades->int, binary->int; apply canonical column order",
            n, n)

    # -- Step 13: Final validation ---------------------------------------------
    print("\n[Step 13] Final dataset validation...")
    remaining_nulls = int(df.isnull().sum().sum())
    invalid_grades  = int(((df[GRADE_COLS] < 0) | (df[GRADE_COLS] > 100)).sum().sum())
    invalid_gpa     = int(((df["gpa"] < 0) | (df["gpa"] > 4)).sum())
    invalid_interest = int((~df["interest"].isin(VALID_INTERESTS)).sum())
    flag_issues = 0
    for key in COURSE_KEYS:
        flag_issues += int(
            ((df[f"{key}_passed"] + df[f"{key}_failed"]) != 1).sum()
        )
    print(f"       Remaining null cells   : {remaining_nulls}")
    print(f"       Invalid grade values   : {invalid_grades}")
    print(f"       Invalid GPA values     : {invalid_gpa}")
    print(f"       Invalid interest values: {invalid_interest}")
    print(f"       Binary flag violations : {flag_issues}")

    validation_passed = (remaining_nulls == 0 and invalid_grades == 0
                         and invalid_gpa == 0 and invalid_interest == 0
                         and flag_issues == 0)
    status = "PASSED OK" if validation_passed else "WARNINGS (see report)"
    print(f"\n       Validation: {status}")
    log.log(13, "Final validation",
            "Check nulls, grade range, GPA range, interest validity, binary flags",
            n, n,
            notes=(f"Nulls={remaining_nulls}, InvalidGrades={invalid_grades}, "
                   f"InvalidGPA={invalid_gpa}, InvalidInterest={invalid_interest}, "
                   f"FlagViolations={flag_issues}"))

    # -- Save outputs ----------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Clean dataset saved  -> {output_path}")
    print(f"     Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # -- Write text report -----------------------------------------------------
    report = build_report(log, initial_rows, initial_cols, len(df), len(df.columns),
                          validation_passed, df)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"[OK] Preprocessing report -> {report_path}")
    print("=" * 64)
    return df


def build_report(log: PreprocessingLog,
                 dirty_rows: int, dirty_cols: int,
                 clean_rows: int, clean_cols: int,
                 passed: bool,
                 df: pd.DataFrame) -> str:
    ts = log.start_time.strftime("%Y-%m-%d %H:%M:%S")
    banner = "=" * 72
    lines = [
        banner,
        "  SMART ACADEMIC ADVISOR — DATA PREPROCESSING REPORT",
        f"  Generated : {ts}",
        banner,
        "",
        "OVERVIEW",
        "--------",
        f"  Input  : data/students_dirty.csv  ({dirty_rows:,} rows × {dirty_cols} columns)",
        f"  Output : data/students.csv         ({clean_rows:,} rows × {clean_cols} columns)",
        f"  Rows removed     : {dirty_rows - clean_rows:,} "
        f"({(dirty_rows - clean_rows) / dirty_rows * 100:.1f}%)",
        f"  Overall status   : {'CLEAN OK' if passed else 'SEE WARNINGS'}",
        "",
        "STEP SUMMARY",
        "------------",
        log.summary_table(),
        "",
        "STEP DETAILS",
        "------------",
        log.detail_lines(),
        "",
        "ISSUES INJECTED (for reference)",
        "--------------------------------",
        "  1.  Missing / null values           GPA ~10%, grades ~8% each",
        "  2.  Duplicate rows                  ~5% of dataset",
        "  3.  Inconsistent interest casing    ai/ml, Ai/Ml, AI ML",
        "  4.  Typos in categorical values     Neetworks, Softwre Eng",
        "  5.  Outlier grades                  -999, 150 (outside 0-100)",
        "  6.  Outlier GPA                     -1.5, 9.99 (outside 0-4)",
        "  7.  Grade as string                 '85%', 'seventy', 'N/A'",
        "  8.  GPA as string                   '3.45a', 'N/A', '  2.7  '",
        "  9.  Inconsistent student_id format  stu-0001, STU 0001, 1",
        "  10. Whitespace / special characters in interest, student_id",
        "  11. None encoding variants          'none','N/A','na','-',''",
        "  12. Binary flag corruption          passed+failed both 0 or 1",
        "  13. Count field inconsistencies     passed_count, failed_count",
        "  14. Float where int expected        grades as 85.7",
        "  15. Junk column                     'extra_info'",
        "",
        "CLEANING STRATEGIES APPLIED",
        "---------------------------",
        "  Missing values    -> Filled with column median (GPA and each grade).",
        "                      Categorical missing -> default 'Undecided'.",
        "  Duplicates        -> Exact-row duplicates dropped; ID-collision",
        "                      duplicates deduplicated (keep first).",
        "  String grades     -> Regex strip of '%', 'pts', 'points'; word-to-",
        "                      number map ('seventy' -> 70); leading numeric",
        "                      extraction; 'N/A' -> NaN.",
        "  String GPA        -> Remove non-numeric chars, cast to float;",
        "                      outliers (< 0 or > 4) nullified then median-filled.",
        "  Outlier grades    -> Values outside [0, 100] set to NaN then filled.",
        "  Interest typos    -> Strip special chars, lowercase, whitespace",
        "                      collapse, map via canonical dictionary.",
        "  Binary flags      -> Recomputed deterministically from cleaned grades",
        "                      (threshold 60). Eliminates all corruption types.",
        "  Summary counts    -> Recomputed from cleaned binary flags.",
        "  None encoding     -> Mapped '', 'n/a', '-', 'null' -> 'None'.",
        "  Junk columns      -> Dropped any column not in the expected schema.",
        "  Data types        -> Enforced: GPA->float, grades/flags/counts->int.",
        "",
        "FINAL DATASET PROFILE",
        "---------------------",
    ]
    lines.append(f"  Total students     : {len(df):,}")
    lines.append(f"  Columns            : {len(df.columns)}")
    lines.append(f"  Missing cells      : {df.isnull().sum().sum()}")
    lines.append(f"  GPA range          : {df['gpa'].min():.2f} – {df['gpa'].max():.2f}")
    lines.append(f"  GPA mean           : {df['gpa'].mean():.2f}")
    for col in ["math_grade", "ai_grade", "programming_grade"]:
        lines.append(f"  {col:<26} mean={df[col].mean():.1f}  "
                     f"min={df[col].min()}  max={df[col].max()}")
    lines.append("")
    lines.append("  Track distribution:")
    for track, count in df["recommended_track"].value_counts().items():
        pct = count / len(df) * 100
        lines.append(f"    {track:<30} {count:>5}  ({pct:5.1f}%)")
    lines.append("")
    lines.append("  Interest distribution:")
    for interest, count in df["interest"].value_counts().items():
        pct = count / len(df) * 100
        lines.append(f"    {interest:<30} {count:>5}  ({pct:5.1f}%)")
    lines.append("")
    lines.append(banner)
    return "\n".join(lines)


if __name__ == "__main__":
    run_pipeline()
