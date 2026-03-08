# %% [markdown]
# # Step 3 & 4: Exploratory Data Analysis (EDA) & Feature Engineering
# 
# **Goal:** Visualize raw data patterns and create new predictive features to improve our classification model.
# 
# **Why is this necessary?**
# - EDA helps us understand what factors drive student dropout visually.
# - Feature Engineering is the art of creating new columns based on existing ones to make the machine learning algorithm smarter.
# 
# **What you learn here:**
# - How to visualize distributions and correlations.
# - How to encode categorical features.
# - How to handle imbalanced data.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a visually appealing theme for our plots
sns.set_theme(style="whitegrid", palette="muted")

# %% [markdown]
# ## 3.1 Load the Data
# *How it is done:* Load the previously cleaned data. If the file is not there, we will mock it for this demonstration.

# %%
try:
    df = pd.read_csv('cleaned_student_data.csv')
    print("Cleaned data loaded.")
except FileNotFoundError:
    print("Creating mock cleaned data for demonstration...")
    # Mock data to demonstrate EDA and FE
    np.random.seed(42)
    df = pd.DataFrame({
        'age_at_enrollment': np.random.randint(18, 45, 100),
        'previous_qualification_grade': np.random.uniform(10, 20, 100),
        'scholarship_holder': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'curricular_units_2nd_sem_grade': np.random.uniform(0, 20, 100),
        'target': np.random.choice(['Dropout', 'Graduate'], 100, p=[0.4, 0.6])
    })

# %% [markdown]
# ## 3.2 Visualizing the Target Variable
# *Why is this necessary?* We need to identify if our dataset is unbalanced. An unbalanced set (e.g., 90% Graduates, 10% Dropouts) requires special care (like SMOTE).
# 
# *How it is done:* Use Seaborn's `countplot` to easily visualize.

# %%
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='target', order=['Graduate', 'Dropout'])
plt.title('Distribution of Student Status (Target)', fontsize=14)
plt.ylabel('Number of Students')
plt.xlabel('Status')

# Add exact numbers above the bars
for p in ax.patches:
    ax.annotate(f'\n{int(p.get_height())}', (p.get_x() + 0.4, p.get_height()), ha='center', va='top', color='white', size=12)

plt.show()

# *Insight extracted:* If 'Dropout' is significantly smaller, consider upsampling algorithms during modeling.

# %% [markdown]
# ## 3.3 Visualizing Continuous Variables vs Target
# *Why is this necessary?* To identify if a continuous feature (like grades or age) has a strong predictive power.
# 
# *How it is done:* We use a `boxplot` to see the spread (median, quartiles, outliers).

# %%
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='target', y='curricular_units_2nd_sem_grade')
plt.title('2nd Semester Grades vs Student Dropout Status')
plt.show()

# *Insight extracted:* Look at the median. If Dropouts typically score lower, this feature is highly predictive. If the boxes completely overlap, it's not a strong feature.

# %% [markdown]
# ## 4.1 Feature Engineering: Binning Ages
# *Why is this necessary?* Sometimes categories act better than raw numbers. E.g., segmenting ages into "Traditional College Age", "Adult Learner" might reveal specific dropout trends better than exact age numbers.
# 
# *How it is done:* Using `pd.cut()` to create age groups.

# %%
# Create age buckets
bins = [17, 21, 25, 30, 60]
labels = ['18-21 (Traditional)', '22-25', '26-30', '30+ (Adult Learner)']
df['age_group'] = pd.cut(df['age_at_enrollment'], bins=bins, labels=labels)

print("Created new Feature: age_group")
display(df[['age_at_enrollment', 'age_group']].head())

# %% [markdown]
# ## 4.2 Feature Engineering: Encoding Categorical Variables
# *Why is this necessary?* Machine learning models only understand matrices of numbers, not text like "Graduate" or "18-21 (Traditional)".
# 
# *How it is done:* `pd.get_dummies()` (One-Hot Encoding) or `LabelEncoder` (for Target).

# %%
# Rule of Thumb: Target variable is Label Encoded (0 and 1)
df['target_encoded'] = df['target'].map({'Graduate': 0, 'Dropout': 1})
# We drop 'Enrolled' students if we only care about the binary Dropout vs Graduate.

# For categorical features, One-Hot Encode
df_encoded = pd.get_dummies(df, columns=['age_group', 'scholarship_holder'], drop_first=True)

# Drop redundant or original columns that are no longer needed
df_encoded.drop(columns=['target', 'age_at_enrollment'], inplace=True, errors='ignore')

print("Data is now ready for modeling!")
display(df_encoded.head())

# Save to CSV for the Model Notebook
df_encoded.to_csv('engineered_student_data.csv', index=False)
print("Saved engineered data to 'engineered_student_data.csv'")
