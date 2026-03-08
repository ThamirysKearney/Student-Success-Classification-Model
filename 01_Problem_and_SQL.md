# 1. Problem Definition

**Goal:** Predict student dropout and academic success using a classification model. Early prediction allows educational institutions to intervene and support students at risk of dropping out.

**Why is this necessary?**
Before writing any code, we must clearly define what we are trying to predict (the Target variable) and how it creates value. In this case, predicting "Dropout", "Enrolled", or "Graduate" helps target interventions.

**What insights to extract:**
- What constitutes a "dropout"?
- What actions can be taken if a student is identified as at-risk?

---

# 2. Data Ingestion & SQL Exploration

**Goal:** Understand the raw data structure and perform initial quality checks using SQL.

**Dataset Context:** The dataset contains demographics, socioeconomic factors, and academic performance metrics.

**Why is this necessary?**
SQL is the industry standard for querying databases. Before importing data into Python, we often inspect it in the data warehouse to check its size, data types, and identify obvious issues (like missing values).

## SQL Tasks

### A. Dataset Inspection
*How it is done:* We select a few rows to understand the grain of the data (what one row represents).
```sql
-- View the first 10 rows to understand the columns
SELECT *
FROM student_data
LIMIT 10;
```
*What insights to extract:* Check if one row represents one student. Look at the data types (are they numbers or text?).

### B. Distribution Analysis (Target Variable)
*How it is done:* We count how many students fall into each category of our target variable (`Target`).
```sql
-- Check the class balance of the target variable
SELECT 
    Target, 
    COUNT(*) as student_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM student_data
GROUP BY Target
ORDER BY student_count DESC;
```
*What insights to extract:* Is the dataset imbalanced? If 90% are "Graduates" and 10% are "Dropouts", our model will struggle to identify dropouts without special techniques.

### C. Aggregation (Feature Exploration)
*How it is done:* Group by a categorical feature (like gender or scholarship) to see how it relates to the target.
```sql
-- Does having a scholarship reduce dropout rates?
SELECT 
    Scholarship_holder,
    Target,
    COUNT(*) as student_count
FROM student_data
GROUP BY Scholarship_holder, Target
ORDER BY Scholarship_holder, Target;
```
*What insights to extract:* Identify early trends. If most scholarship holders graduate, this is a strong predictive feature.

### D. Data Quality Checks
*How it is done:* Check for missing or anomalous values in critical columns.
```sql
-- Check for missing or invalid age values
SELECT 
    COUNT(*) as total_rows,
    SUM(CASE WHEN Age_at_enrollment IS NULL THEN 1 ELSE 0 END) as missing_age,
    SUM(CASE WHEN Age_at_enrollment < 16 OR Age_at_enrollment > 80 THEN 1 ELSE 0 END) as invalid_age
FROM student_data;
```
*What insights to extract:* If there are many nulls, we know we will need to handle them in Python (Data Cleaning phase) before training our model.
