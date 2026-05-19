"""
setup.py
One-shot setup for the Smart Academic Advisor.

Pipeline
────────
  1. Install dependencies
  2. Generate DIRTY / UNCLEAN dataset   (generate_dirty_data.py)
  3. Run preprocessing pipeline         (preprocess_data.py)
  4. Train models on clean dataset      (train_models.py)

Run: python setup.py
"""
from __future__ import annotations

import subprocess
import sys


def run(command: list[str]) -> None:
    # Run each setup command and stop immediately if any step fails.
    # بالمصري: لو خطوة من setup فشلت، بنوقف عشان منكمّلش على داتا أو موديل ناقص.
    print("\n$ " + " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(command)}")


print("=" * 64)
print("  Smart Academic Advisor - Setup")
print("=" * 64)

print("\n[1/4] Installing dependencies from requirements.txt...")
run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

print("\n[2/4] Generating DIRTY synthetic dataset (data/students_dirty.csv)...")
run([sys.executable, "generate_dirty_data.py"])

print("\n[3/4] Running preprocessing pipeline -> data/students.csv ...")
run([sys.executable, "preprocess_data.py"])

print("\n[4/4] Training models on clean dataset...")
run([sys.executable, "train_models.py"])

print("\n" + "=" * 64)
print("  Setup complete.")
print("")
print("  Artifacts produced:")
print("    data/students_dirty.csv         Raw / unclean dataset")
print("    data/students.csv               Cleaned dataset")
print("    data/preprocessing_report.txt   Full cleaning log")
print("    models/trained_models.pkl       Trained ML models")
print("")
print("  Run:  python app.py")
print("  Open: http://127.0.0.1:5000")
print("=" * 64)
