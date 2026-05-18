# Smart Academic Advisor - Full Project Documentation & Study Guide

## 1. Overview and Core Philosophy
The **Smart Academic Advisor** is a comprehensive, end-to-end web application that uses Machine Learning to simulate an intelligent academic advisory system. Given the privacy restrictions surrounding real university data, this project employs a synthetic data generation and cleaning pipeline to create realistic scenarios.

The primary goal of the system is to analyze a student's:
- **Academic Performance**: Overall GPA and 8 core course grades.
- **Course History**: Which courses have been passed, failed, and need retaking.
- **Interests**: The student's primary academic interest (e.g., AI/ML, Networks).

Using this data, the application recommends a specific academic track, provides evidence for the recommendation, flags potential academic risks, and outlines a graduation path.

---

## 2. Study Guide & Learning Objectives
If you are studying this project to understand Data Mining, Machine Learning, or Web Development, follow this reading path:

1. **Understand the Data Problem**: Read `generate_dirty_data.py` to see how synthetic data is created and deliberately corrupted to mimic real-world datasets.
2. **Master ETL (Extract, Transform, Load)**: Read `preprocess_data.py`. This is the core Data Mining component where messy data is wrangled, standardized, and made suitable for ML algorithms.
3. **Explore the Machine Learning Models**: Read `train_models.py` to understand how Scikit-Learn is used to train Decision Trees, Random Forests, and KNN models, and how Apriori is used for association rule mining.
4. **Trace the Web Integration**: Read `app.py` to see how the trained models are deserialized and exposed as a web service via Flask.

---

## 3. File-by-File Code & Functionality Breakdown

This section breaks down the core Python scripts, explaining their code logic, functionality, and the specific algorithms used.

### A. `generate_dirty_data.py` (Phase 1: Data Generation)
**Purpose**: Synthesizes a realistic student dataset and intentionally corrupts it to simulate real-world data mining problems.
**Libraries Used**: `pandas`, `numpy`, `random`.

**Code Logic & Functions**:
1. **`generate_clean_base()`**: Creates 900 base records. It uses weighted heuristics. For example, if a student's assigned track is "AI/Machine Learning", their grades in `Mathematics` and `AI` are forced to be higher using `numpy.random.normal()` distributions.
2. **`inject_missing_values()`**: Randomly selects cells across the DataFrame (e.g., in GPA or grades) and sets them to `np.nan`. Simulates data entry failures.
3. **`inject_duplicates()`**: Randomly samples existing rows and concatenates them to the DataFrame to simulate duplicated database records.
4. **`inject_outliers()`**: Corrupts numerical data by multiplying GPAs by 10 (e.g., 35.0 instead of 3.5) or creating negative grades (-15).
5. **`inject_type_errors()`**: Converts integers into strings (e.g., turning the integer `85` into the string `"85%"` or `"eighty-five"`).

**Execution Outcome**: Outputs `data/students_dirty.csv` with exactly 945 rows containing 15 distinct data quality issues.

---

### B. `preprocess_data.py` (Phase 2: Data Mining & Cleaning)
**Purpose**: The core ETL pipeline that ingests the dirty data, cleanses it using data mining techniques, and produces a mathematically sound dataset.
**Libraries Used**: `pandas`, `numpy`, `re` (Regex).

**Code Logic & Functions**:
1. **Drop Junk Columns**: `df.drop(columns=['extra_info'])` removes noise columns that have zero variance or predictive value.
2. **Deduplication**: `df.drop_duplicates(keep='first')` removes the duplicate rows injected in Phase 1, bringing the row count back to 900.
3. **Regex String Parsing (`clean_grade_strings`)**: Uses regular expressions (`re.sub(r'[^\d.]', '', str_val)`) to strip out percent signs or letters from grade columns so they can be cast back to `float`.
4. **Outlier Nullification & Imputation**: 
   - **Logic**: Any GPA `< 0` or `> 4.0` is converted to `NaN`. Any grade `< 0` or `> 100` is converted to `NaN`.
   - **Imputation**: `df['gpa'].fillna(df['gpa'].median())` replaces the `NaNs` with the median value of the column, which is robust against remaining outliers compared to the mean.
5. **Standardization (`standardize_interest`)**: Uses dictionary mapping to fix case-sensitivity and typos (e.g., mapping `"ai/ml"`, `"AI_ML"`, and `"Artificial Intelligence"` to a single `"AI/ML"` category).
6. **Feature Engineering / Constraint Checking**: Recalculates binary columns (e.g., `passed_course_X`) by checking `if grade >= 50`. This ensures logical consistency if the grade was altered during imputation.

**Execution Outcome**: Outputs `data/students.csv` (100% clean) and `data/preprocessing_report.txt`.

---

### C. `train_models.py` (Phase 3: Machine Learning)
**Purpose**: Trains classification algorithms to predict the best academic track and mines association rules.
**Libraries Used**: `scikit-learn` (DecisionTree, RandomForest, KNN, StandardScaler, LabelEncoder), `mlxtend` (Apriori).

**Code Logic & Functions**:
1. **Data Preparation**:
   - **Feature Matrix (`X`)**: Extracts the GPA, 8 course grades, and the categorical `Interest` column.
   - **Target Vector (`y`)**: The `recommended_track` column.
   - **Encoding**: Uses `LabelEncoder` to convert the string `Interest` and `recommended_track` columns into integers so the math algorithms can process them.
   - **Scaling**: Uses `StandardScaler` to normalize numerical data (GPA is 0-4, grades are 0-100). This prevents KNN from being biased toward the larger grade numbers.
2. **Model Training & Evaluation**:
   - `DecisionTreeClassifier(max_depth=5)`: Trained to provide simple, interpretable tree structures.
   - `RandomForestClassifier(n_estimators=100)`: Trains an ensemble of 100 decision trees to ensure high accuracy and output probability scores (Confidence %).
   - `KNeighborsClassifier(n_neighbors=5)`: Trained to find the 5 closest historical students based on Euclidean distance in the scaled feature space.
   - `cross_val_score()`: Runs 5-fold cross-validation to ensure the models aren't overfitting.
3. **Association Rule Mining**:
   - Converts the passed courses and tracks into a transactional format (one-hot encoding).
   - `apriori(min_support=0.1)`: Finds frequent itemsets (e.g., passing AI and passing DB).
   - `association_rules(metric="lift", min_threshold=1.2)`: Extracts the final rules that prove a strong correlation.

**Execution Outcome**: Serializes all trained models, encoders, and the scaler into `models/trained_models.pkl` using the `pickle` library.

---

### D. `app.py` (Phase 4: Flask Web Application)
**Purpose**: Exposes the trained models as a web service and renders the HTML UI.
**Libraries Used**: `flask` (Flask, request, render_template), `pickle`, `numpy`.

**Code Logic & Routes**:
1. **Initialization**: During app startup, it loads `models/trained_models.pkl` into memory so prediction requests are instantaneous.
2. **`@app.route('/')`**: Renders `index.html`, the landing page.
3. **`@app.route('/advisor')`**: Renders the input form. It passes the available `Interest` categories to the template so the dropdown is dynamically populated.
4. **`@app.route('/predict', methods=['POST'])`**: The core logic controller:
   - Extracts form data from `request.form`.
   - Constructs a 2D numpy array representing the single student: `[[gpa, grade1, grade2... encoded_interest]]`.
   - Calls `scaler.transform()` to normalize the single input based on the training data's distribution.
   - Calls `model.predict_proba()` (using Random Forest) to get the recommended track and its confidence percentage.
   - Calls `knn.kneighbors()` to fetch the 5 most similar historical students.
   - Renders `result.html` passing all predictions, warnings (e.g., if a grade < 50), and KNN data to the Jinja2 template engine.
5. **`@app.route('/dashboard')`**: Aggregates the global metrics (Cross-validation scores, feature importances, extracted Apriori rules) and passes them to `dashboard.html` for visualization.

---

## 4. UI/UX and Frontend Implementation
- **Jinja2 Templating**: The application uses Flask's built-in Jinja2 engine (`{% if %}`, `{% for %}`) to dynamically generate HTML tables and warning banners based on the model's output.
- **CSS Architecture**: `static/css/style.css` contains all the styling. It implements CSS Variables (`--primary-color`, `--bg-color`) to support a clean, modern aesthetic and an easy-to-implement Dark Mode toggle via JavaScript.
- **No External Dependencies**: To maintain lightweight performance, the Dashboard charts are constructed using native HTML `<div>` elements with dynamic CSS widths, avoiding the need for heavy libraries like Chart.js.
