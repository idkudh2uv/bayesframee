# run_framework.py
import pandas as pd
import os
import numpy as np
from bayesframe import BayesFrame
from utilities import load_test_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import beta
import matplotlib.pyplot as plt

# --- Configuration ---
TARGET_COLUMN = 'Visibility (km)'
OUTPUT_CSV_PATH = 'predictions_output.csv'
# --- End Configuration ---

print("Loading dataset...")
training_df = pd.read_csv("data/weatherhistory.csv")

print("Preparing data...")
training_df = training_df.dropna()  # Drop NA values

# Remove non-numerical columns
columns_to_remove = ['Formatted Date', 'Summary', 'Precip Type', 'Daily Summary']
for col in columns_to_remove:
    if col in training_df.columns:
        training_df = training_df.drop(col, axis=1)

# Split data
train_df, test_df = train_test_split(training_df, test_size=0.2, random_state=42)

# --- BayesFrame (LinZoo) ---
print("Initializing BayesFrame and building model zoo...")
bf = BayesFrame(df=train_df, target=TARGET_COLUMN, model_scheme=['averaging', 'all'])

# --- BMA Point Prediction ---
print("Generating BMA point predictions...")
bma_predictions = bf(data=test_df, target=TARGET_COLUMN, outpath=None, print_rmse=False) #get the predictions.

# --- Enhanced Beta Distribution ---
print("Enhanced Beta distribution...")
test_df['y_binary'] = (test_df[TARGET_COLUMN] == 10).astype(int)

if test_df['y_binary'].nunique() > 1:
    logreg = LogisticRegression()
    logreg.fit(test_df.drop(columns=[TARGET_COLUMN, 'y_binary']), test_df['y_binary'])
    prob_y10 = logreg.predict_proba(test_df.drop(columns=[TARGET_COLUMN, 'y_binary']))[:, 1]
else:
    prob_y10 = test_df['y_binary'].mean()

visibility_lt_10 = test_df[test_df[TARGET_COLUMN] < 10][TARGET_COLUMN]
visibility_scaled = visibility_lt_10 / 10.0

if len(visibility_scaled) > 1 and visibility_scaled.var() > 0:
    mean = visibility_scaled.mean()
    variance = visibility_scaled.var()
    alpha = mean * (mean * (1 - mean) / variance - 1)
    beta_param = (1 - mean) * (mean * (1 - mean) / variance - 1)
    beta_predictions = beta.ppf(0.5, alpha, beta_param) * 10.0
else:
    beta_predictions = visibility_lt_10.median() if not visibility_lt_10.empty else 0.0

final_predictions = np.where(test_df['y_binary'] == 1, 10.0, beta_predictions)

# --- Evaluation ---
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test_df[TARGET_COLUMN], final_predictions))
print(f"RMSE: {rmse}")

# --- Visualization (Optional) ---
plt.figure(figsize=(10, 6))
plt.scatter(test_df[TARGET_COLUMN], final_predictions)
plt.xlabel("Actual Visibility")
plt.ylabel("Predicted Visibility")
plt.title("Actual vs. Predicted Visibility")
plt.show()

print("\nFramework run complete.")
