# Smart Academic Advisor - Full Project Documentation, Study Guide, & Defense Q&A

## 1. Overview and Core Philosophy
The **Smart Academic Advisor** is a comprehensive, end-to-end web application that uses Machine Learning to simulate an intelligent academic advisory system. Given the privacy restrictions surrounding real university data, this project employs a synthetic data generation and cleaning pipeline to create realistic scenarios.

The primary goal of the system is to analyze a student's:
- **Academic Performance**: Overall GPA and 8 core course grades.
- **Course History**: Which courses have been passed, failed, and need retaking.
- **Interests**: The student's primary academic interest (e.g., AI/ML, Networks).

Using this data, the application recommends a specific academic track, provides evidence for the recommendation, flags potential academic risks, and outlines a graduation path.

---

## 2. Project Defense & Discussion Q&A Guide
If you are presenting this project, expect the following questions from your professors or peers. Memorize these answers to defend your architectural and algorithmic choices confidently.

### Q1: Why did you use Synthetic Data instead of Real University Data?
**Answer**: University academic records are strictly protected under privacy laws (like FERPA). Since we could not legally acquire real student transcripts for this project, we built a synthetic data generator (`generate_dirty_data.py`). The generator uses realistic statistical heuristics (e.g., forcing a correlation between high math grades and the AI track) to ensure the Machine Learning models have actual, logical patterns to discover, rather than just learning random noise. If real data becomes available, the pipeline is fully ready to ingest it.

### Q2: Why did you intentionally create "Dirty" Data?
**Answer**: In the real world, Data Mining is 80% data cleaning and 20% modeling. By intentionally injecting 15 different types of data quality issues (missing values, duplicates, impossible outliers like a GPA of 9.9, and string typos like "85%"), we demonstrated a robust **ETL (Extract, Transform, Load)** pipeline. It proves we can handle raw, messy datasets and mathematically clean them (`preprocess_data.py`) before feeding them into our models.

### Q3: How do you handle Missing Values (NaNs)? Why not just delete those rows?
**Answer**: If we deleted every row with a missing value, we would lose too much data. Instead, we use **Imputation**. For numerical fields like GPA, we replace missing values with the **Median**. We chose the median over the mean because the median is mathematically robust against extreme outliers. For categorical fields like `Interest`, we use the **Mode** (the most frequent value).

### Q4: Why did you train three different models (Decision Tree, Random Forest, KNN)?
**Answer**: Each model serves a different purpose in demonstrating our Data Mining knowledge:
- **Decision Tree**: Provides interpretability. We can actually visualize the exact "if-then" rules the model uses to make a decision.
- **Random Forest**: This is our primary engine. By building an ensemble of many decision trees, it reduces overfitting and provides the highest accuracy, as well as a "Confidence Percentage" for the prediction.
- **K-Nearest Neighbors (KNN)**: We use this to find "Similar Historical Students". It groups students by geometric distance, allowing us to show the user real examples of past students who had similar grades and what tracks they succeeded in.

### Q5: Why do you use `StandardScaler` before prediction?
**Answer**: Algorithms like KNN calculate the Euclidean distance between data points. Our GPA is on a scale of 0.0 to 4.0, while grades are 0 to 100. Without scaling, the KNN algorithm would think the grades are drastically more important than the GPA just because the numbers are bigger. `StandardScaler` normalizes all features to have a mean of 0 and a standard deviation of 1, ensuring every feature is treated fairly.

### Q6: What is the purpose of the Apriori Algorithm (Association Rules) in this project?
**Answer**: While Random Forest predicts the track, Apriori performs unsupervised learning to discover hidden relationships (frequent itemsets). For example, it might find a rule saying: `If a student passes Algorithms and AI -> They have an 85% probability of choosing the AI Track`. We use these rules to provide supplementary, data-backed evidence in the advisor's final report.

### Q7: How do you ensure your models aren't overfitting?
**Answer**: We use **K-Fold Cross-Validation** (specifically 5-fold) during the training phase. Instead of just splitting the data into one training and one testing set, the model is trained and tested 5 different times on different subsets of the data. The average accuracy from these 5 folds is what we report on the dashboard, proving the model generalizes well to unseen data.

---

## 3. Phase 1: Synthetic Data Generation (`generate_dirty_data.py`)
### The Process:
- **Base Records**: Generates 900 perfectly formatted base records based on realistic academic heuristics.
- **Data "Dirtying" (15 Issue Types)**: Injects anomalies to increase the row count to 945. It introduces:
  - Missing values (NaNs) in GPA, grades, and interests.
  - Exact duplicate rows.
  - Outliers (e.g., negative grades, GPAs over 4.0).
  - Type errors (e.g., string representations of numbers like "85%").
  - Inconsistent formatting (e.g., "stu-001" vs "STU 001").

---

## 4. Phase 2: Data Mining & Preprocessing (`preprocess_data.py`)
### The Cleaning Pipeline:
1. **Column Filtering**: Drops irrelevant "junk" columns (`extra_info`).
2. **Deduplication**: Identifies and removes the duplicate rows injected in Phase 1.
3. **Regex String Parsing**: Uses regular expressions (`re.sub`) to strip out percent signs or letters from grade columns so they can be cast to `float`.
4. **Outlier Nullification**: Any GPA `< 0` or `> 4.0` is converted to `NaN`. Any grade `< 0` or `> 100` is converted to `NaN`.
5. **Imputation**: Fills missing numerical values using statistical medians.
6. **Standardization**: Uses dictionary mapping to fix case-sensitivity and typos (e.g., mapping `"ai/ml"`, `"AI_ML"` to `"AI/ML"`).
7. **Constraint Checking**: Recalculates binary pass/fail columns to ensure logical consistency.

---

## 5. Phase 3: Machine Learning Training (`train_models.py`)
### Code Logic & Algorithms:
1. **Data Preparation**: Extracts the feature matrix (`X`) and target vector (`y`). Uses `LabelEncoder` to convert strings to integers, and `StandardScaler` to normalize numerical data.
2. **Model Training**:
   - `DecisionTreeClassifier(max_depth=5)`: Prevents infinite tree depth to avoid overfitting.
   - `RandomForestClassifier(n_estimators=100)`: Ensemble of 100 trees for highest accuracy.
   - `KNeighborsClassifier(n_neighbors=5)`: Fetches the 5 geometrically closest neighbors.
3. **Association Rules**: Uses `mlxtend`'s `apriori(min_support=0.1)` and `association_rules(metric="lift", min_threshold=1.2)` to extract correlations between passed courses and final tracks.

---

## 6. Phase 4: Web Application Architecture (`app.py`)
### The Controller Logic:
- **`/advisor`**: Renders the input form.
- **`/predict`**: The core logic controller. It extracts form data, constructs a 2D numpy array (`[[gpa, grade1...]]`), calls `scaler.transform()` to normalize it, and calls `model.predict_proba()` to get the recommended track and confidence. It also fetches the `knn.kneighbors()`.
- **`/dashboard`**: Aggregates the global metrics (Cross-validation scores, feature importances, extracted Apriori rules) and passes them to `dashboard.html` for visualization.

### UI/UX Implementation:
- Uses Flask's built-in **Jinja2 engine** (`{% if %}`, `{% for %}`) to dynamically generate HTML tables and warning banners based on the model's output.
- Features a **Dark Mode** toggle and print-friendly CSS for saving the advisor reports to PDF. No heavy charting libraries are used; all dashboard visuals are native HTML/CSS for extreme speed.
