# %% [markdown]
# # Step 2: Data Ingestion and Cleaning
# 
# **Goal:** Load the dataset and handle missing values, incorrect data types, and duplicates before moving to analysis.
# 
# **Why is this necessary?** 
# Real-world data is messy. If we pass "bad" data to a machine learning model, it will learn bad patterns (Garbage In, Garbage Out).
# 
# **What you learn here:**
# - How to load CSV files.
# - How to validate the loaded data.
# - How to check for missing/null values and duplicates.
# - Why dropping or filling missing data is important.

# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 2.1 Data Ingestion
# First, we load the dataset. We assume the Kaggle CSV is downloaded as `data.csv`.
# 
# *How it is done:* We use `pd.read_csv()` from the pandas library, which is the standard tool for data manipulation in Python.

# %%
# Load the dataset
# Replace 'dataset.csv' with the actual file path after downloading from Kaggle
try:
    df = pd.read_csv('dataset.csv', sep=';') # Sometimes Kaggle datasets use ';' instead of ','
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Dataset not found. Please ensure 'data.csv' is in the same folder.")
    # Creating dummy data for demonstration purposes if the file is missing
    df = pd.DataFrame({
        'Marital status': [1, 2, 1, 1, 4],
        'Course': [171, 9254, 9070, 9773, 8014],
        'Age at enrollment': [20, 19, 21, 22, 18],
        'Target': ['Dropout', 'Graduate', 'Dropout', 'Enrolled', 'Graduate']
    })

# %% [markdown]
# ## 2.2 Initial Data Validation & Exploration
# *How it is done:* We look at the first few rows, the shape (rows and columns), and data types.
# 
# *Why is this necessary?* To ensure Pandas read the numbers as numbers, and text as text.

# %%
# View the first 5 rows
display(df.head())

# Learn about the data types and non-null counts
df.info()

# %% [markdown]
# *Insights from `df.info()`:* Let's look if there are missing values (non-null count < total rows) and if numeric fields are correctly inferred as `int64` or `float64`. The `Target` column should be an `object` (text).

# %% [markdown]
# ## 2.3 Checking for Missing Values
# *How it is done:* We sum up all the missing values for each column using `.isnull().sum()`.
# 
# *Why is this necessary?* Machine learning algorithms like Random Forest or Logistic Regression cannot train on data with missing fields.

# %%
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values[missing_values > 0])

# If we had missing values, we could drop the rows (if very few) or impute them (fill them with the mean or median).
# Example: df['Age'].fillna(df['Age'].median(), inplace=True)

# %% [markdown]
# ## 2.4 Checking for Duplicates
# *Why is this necessary?* Duplicate rows can artificially inflate the importance of certain patterns and skew our model.
# 
# *How it is done:* Use `.duplicated().sum()`.

# %%
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows found: {duplicates}")

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates removed.")

# %% [markdown]
# ## 2.5 Normalizing Column Names (Good Practice)
# *Why is this necessary?* Columns with spaces or uppercase letters can be annoying to type. Let's make them lowercase and use underscores.
# 
# *How it is done:* using a list comprehension or `.str.lower()`.

# %%
# Clean column names by replacing spaces with underscores and making them lowercase
df.columns = df.columns.str.lower().str.replace(' ', '_')
print("Cleaned column names:", df.columns.tolist()[:5])

# %% [markdown]
# ## Save the Cleaned Data
# Now that our data is clean, we save it as a new CSV so we don't have to repeat these steps later.

# %%
# Save the cleaned dataset for the next step 
df.to_csv('cleaned_student_data.csv', index=False)
print("Cleaned data saved as 'cleaned_student_data.csv'")
