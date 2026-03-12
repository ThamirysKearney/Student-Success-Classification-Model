# %% [markdown]
# # Step 5: Model Training, Evaluation & Interpretability
# 
# **Goal:** Train a classification algorithm (Random Forest) to predict dropouts and evaluate its performance.
# 
# **Why is this necessary?**
# - The model encapsulates all the data patterns into a mathematical function.
# - We must ensure it predicts accurately before making business decisions based on it.
# 
# **What you learn here:**
# - Preprocessing pipelines (Train-test split, Scaling).
# - How to train a Random Forest classifier.
# - How to interpret Accuracy, Recall, and the Confusion Matrix.
# - Feature Importance (Model Interpretability).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %% [markdown]
# ## 5.1 Prepare the Data (Train-Test Split)
# *Why is this necessary?* If we evaluate the model on the data it trained on, it's like giving a student the actual exam questions beforehand. We split the data: 80% to train (study), 20% to test (exam).
# 
# *How it is done:* `train_test_split` from `sklearn`.

# %%
try:
    df = pd.read_csv('engineered_student_data.csv')
    print("Engineered Data Loaded.")
except FileNotFoundError:
    print("Creating mock engineered data...")
    np.random.seed(42)
    df = pd.DataFrame({
        'curricular_units_2nd_sem_grade': np.random.uniform(0, 20, 100),
        'previous_qualification_grade': np.random.uniform(10, 20, 100),
        'target_encoded': np.random.choice([0, 1], 100, p=[0.6, 0.4])
    })

# Define Features (X) and Target (y)
X = df.drop(columns=['target_encoded'])
y = df['target_encoded']

# Split the Data: 80% Train, 20% Test. We use random_state=42 for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Testing shapes: X={X_test.shape}, y={y_test.shape}")

# %% [markdown]
# ## 5.2 Model Training
# *Why is this necessary?* This is the machine learning part! A Random Forest builds multiple decision trees and averages them to predict whether a student will dropout or not. It's resistant to overfitting and easy to use.
# 
# *How it is done:* Instantiate the model and call `.fit()`.

# %%
# Instantiate model. 100 trees, reproducible state.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train ("Learn" from the data)
rf_model.fit(X_train, y_train)
print("Model Training Completed!")

# %% [markdown]
# ## 5.3 Model Evaluation
# *Why is this necessary?* To answer "Is the model actually good?". In classification, we look at several metrics. For predicting dropouts, **Recall** is crucial (catching all the actual dropouts, even if we flag a few false positives).
# 
# *How it is done:* Generate predictions on Test data, compare to Truth.

# %%
# Generate predictions
y_pred = rf_model.predict(X_test)

# Calculate Accuracy (Total correct / Total predictions)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2%}\n")

# Print the Classification Report
# Focus on Recall for Class 1 (Dropout)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Graduate (0)', 'Dropout (1)']))

# %% [markdown]
# *Insights extracted:*
# If Precision for Dropout is 0.80, it means 80% of those we flagged as Dropouts actually dropped out.
# If Recall for Dropout is 0.60, it means we only successfully identified 60% of the true dropouts in the dataset. Our business goal is to maximize Recall without destroying Precision.

# %% [markdown]
# ## 5.4 The Confusion Matrix
# *How it is done:* Visualizing False Positives and False Negatives.

# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Graduate', 'Dropout'], yticklabels=['Graduate', 'Dropout'])
plt.ylabel('Actual Truth')
plt.xlabel('Model Prediction')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# ## 5.5 Model Interpretability (Feature Importance)
# *Why is this necessary?* Educational institutions need to know *why* a student is flagged to intervene effectively. Black box models don't tell you why. Feature Importance ranks what variables mattered most to the model.
# 
# *How it is done:* Extract `feature_importances_` from the trained Random Forest.

# %%
# Extract importances and map them to column names
importances = rf_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title("What Drives Dropouts? (Feature Importance)")
plt.xlabel("Importance Score")
plt.show()

# *Insight extracted:* The top feature (e.g., Curricular Units 2nd Sem Grade) should be the core focus of the school's intervention program.
