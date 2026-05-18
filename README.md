# Smart Academic Advisor Application

A complete academic advising web application built with Flask and machine learning. The system recommends a suitable academic track, elective courses, and a graduation path by analyzing academic performance, passed and failed courses, and student interests.

## Requirement Coverage

| Requirement | Implementation |
|---|---|
| Academic performance | GPA and 8 course grades are used by the models. |
| Passed and failed courses | Pass/fail indicators, failed count, warnings, and retake plan are included. |
| Student interests | Interest is included as a categorical ML feature and in association rules. |
| Decision Tree | Implemented as a trained classification pipeline. |
| Random Forest | Implemented as the recommended high-accuracy model. |
| KNN | Implemented with feature scaling and a similar-students section. |
| Association Rules | Apriori rules are mined and matched to each student profile. |
| Outcome | Provides track, confidence, electives, career paths, warnings, and graduation path. |
| Interface | Modern Flask web interface with clean responsive UI and dark mode. |

## Dataset & Preprocessing Pipeline

The project generates its own synthetic academic advising dataset and features a comprehensive data cleaning pipeline simulating real-world data mining scenarios.

1. **Dirty Data Generation**: `generate_dirty_data.py` generates 945 messy student records containing 15 distinct types of data quality issues.
2. **Preprocessing**: `preprocess_data.py` cleans the messy dataset, resolves missing values and outliers, removes duplicates, and outputs 900 clean records along with a `preprocessing_report.txt`.

The final clean dataset (`data/students.csv`) contains:
- student ID, GPA, 8 course grades, selected interest
- passed/failed courses and counts
- binary pass/fail indicators
- recommended academic track

## Full Project Details
For a deep dive into the architecture, ML workflow, and UI/UX design, please read the [Full Project Documentation](PROJECT_DOCUMENTATION.md).

## Quick Start (Recommended)

To run the full pipeline (data generation -> preprocessing -> model training -> web app):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run automated setup and launch application
python setup.py
python app.py
```

Open your browser to: `http://127.0.0.1:5000`

## Project Structure

```text
smart_academic_advisor/
├── app.py                      # Flask Application
├── generate_dirty_data.py      # Synthesizes messy dataset
├── preprocess_data.py          # Data cleaning pipeline
├── train_models.py             # Trains Scikit-Learn models
├── setup.py                    # Automated end-to-end setup script
├── run_checks.py               # Route logic testing
├── requirements.txt
├── README.md
├── PROJECT_DOCUMENTATION.md    # Full documentation
├── data/
│   ├── students_dirty.csv      # Raw / unclean dataset
│   ├── students.csv            # Cleaned dataset
│   └── preprocessing_report.txt# Cleaning log
├── models/
│   └── trained_models.pkl      # Trained ML models
├── templates/                  # HTML Templates
└── static/                     # CSS / Images
```

## Recommended Demo Steps

1. Run the app and open the Advisor page.
2. Click one of the demo sample buttons to fill the form quickly.
3. Generate the advisor report and explore the recommended track, evidence, warnings, electives, and similar students.
4. Navigate to the Dashboard to review model performance, feature importance, confusion matrices, and association-rule patterns.
5. Review `data/preprocessing_report.txt` to observe the data cleaning process.

## Testing

Run:

```bash
python run_checks.py
```

Expected result:

```text
All route and UI checks passed.
```
