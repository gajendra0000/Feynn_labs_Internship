# Credit Risk Modeling Project

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, r2_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

## Part 1: Data Generation

print("\n--- Part 1: Data Generation ---")

# Number of samples
n_samples = 2000

# Demographics / Application Variables
age = np.random.randint(22, 65, n_samples)
annual_income = np.random.normal(loc=70000, scale=25000, size=n_samples).astype(int)
annual_income[annual_income < 20000] = 20000 # Minimum income
emp_length = np.random.choice([f'{i} years' for i in range(1, 11)] + ['< 1 year', '10+ years'], n_samples)
home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.4, 0.2, 0.4])
loan_amount = np.random.normal(loc=15000, scale=7000, size=n_samples).astype(int)
loan_amount[loan_amount < 1000] = 1000 # Minimum loan amount
purpose = np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase'], n_samples, p=[0.4, 0.3, 0.1, 0.1, 0.1])
loan_term = np.random.choice([' 36 months', ' 60 months'], n_samples, p=[0.7, 0.3])
verification_status = np.random.choice(['Verified', 'Source Verified', 'Not Verified'], n_samples, p=[0.35, 0.35, 0.3])

# Behavioral / Credit History
revol_util = np.random.normal(loc=0.4, scale=0.2, size=n_samples)
revol_util[revol_util < 0] = 0
revol_util[revol_util > 1] = 1
revol_bal = np.random.normal(loc=15000, scale=10000, size=n_samples).astype(int)
revol_bal[revol_bal < 0] = 0
earliest_credit_line = [datetime.now() - timedelta(days=np.random.randint(365 * 3, 365 * 20)) for _ in range(n_samples)]
credit_inquiries_last_6m = np.random.randint(0, 6, n_samples)
open_credit_lines = np.random.randint(2, 15, n_samples)
credit_utilization = np.random.normal(loc=0.5, scale=0.2, size=n_samples)
credit_utilization[credit_utilization < 0] = 0
credit_utilization[credit_utilization > 1] = 1
delinquency_2yrs = np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.07, 0.03])
total_current_balance = np.random.normal(loc=50000, scale=30000, size=n_samples).astype(int)
total_current_balance[total_current_balance < 0] = 0

# Target Variables
loan_status = np.random.choice(['fully_paid', 'charged_off', 'default'], n_samples, p=[0.7, 0.2, 0.1])
recovered_amount = np.where(loan_status != 'fully_paid', np.random.uniform(0, loan_amount * 0.5, n_samples), 0)
total_rec_prncp = np.where(loan_status == 'fully_paid', loan_amount, np.random.uniform(0, loan_amount * 0.8, n_samples))
funded_amnt = loan_amount # For simplicity, assume funded_amnt is same as loan_amount for now

# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'annual_income': annual_income,
    'emp_length': emp_length,
    'home_ownership': home_ownership,
    'loan_amount': loan_amount,
    'purpose': purpose,
    'loan_term': loan_term,
    'verification_status': verification_status,
    'revol_util': revol_util,
    'revol_bal': revol_bal,
    'earliest_credit_line': earliest_credit_line,
    'credit_inquiries_last_6m': credit_inquiries_last_6m,
    'open_credit_lines': open_credit_lines,
    'credit_utilization': credit_utilization,
    'delinquency_2yrs': delinquency_2yrs,
    'total_current_balance': total_current_balance,
    'loan_status': loan_status,
    'recovered_amount': recovered_amount,
    'total_rec_prncp': total_rec_prncp,
    'funded_amnt': funded_amnt
})

# Save to CSV
data.to_csv('credit_risk_synthetic.csv', index=False)

print('Synthetic dataset generated and saved to credit_risk_synthetic.csv')
print(data.head())

## Part 2: Preprocessing & Feature Engineering

print("\n--- Part 2: Preprocessing & Feature Engineering ---")

# Load the dataset
df = pd.read_csv("credit_risk_synthetic.csv")

# Convert emp_length to numerical
def emp_length_to_int(emp_len):
    if pd.isna(emp_len): return 0
    if isinstance(emp_len, (int, float)): return emp_len # Already converted
    if isinstance(emp_len, str):
        if '10+ years' in emp_len: return 10
        if '< 1 year' in emp_len: return 0
        return int(emp_len.replace(" years", ""))
    return 0 # Default for unexpected types
df["emp_length"] = df["emp_length"].apply(emp_length_to_int)

# Convert loan_term to numerical
df["loan_term"] = df["loan_term"].str.replace(" months", "").astype(int)

# Convert earliest_credit_line to datetime and calculate months_since_earliest_credit
df["earliest_credit_line"] = pd.to_datetime(df["earliest_credit_line"])
df["issue_date"] = datetime.now() # Assuming loans are issued now for simplicity
df["months_since_earliest_credit"] = ((df["issue_date"] - df["earliest_credit_line"]).dt.days / 30).astype(int)

# Drop issue_date and earliest_credit_line as they are no longer needed
df = df.drop(columns=["issue_date", "earliest_credit_line"])

# Missing Value Treatment (using median for numeric, mode for categorical)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# Derived Features
df["installment_to_income"] = df["loan_amount"] / df["annual_income"]
# For credit_limit, we need to simulate it as it's not in the original synthetic data
df["credit_limit"] = df["revol_bal"] / df["revol_util"]
df["credit_limit"] = df["credit_limit"].replace([np.inf, -np.inf], np.nan).fillna(df["revol_bal"] * 2) # Handle division by zero or very small revol_util
df["credit_utilization_derived"] = df["revol_bal"] / df["credit_limit"]

# Replace infinite values with NaN before binning
for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# Feature Binning (fine classing via quantiles for continuous variables)
# Select continuous variables for binning
continuous_vars = ["age", "annual_income", "loan_amount", "revol_util", "revol_bal", 
                   "credit_inquiries_last_6m", "open_credit_lines", "credit_utilization", 
                   "delinquency_2yrs", "total_current_balance", "installment_to_income", 
                   "months_since_earliest_credit", "credit_limit", "credit_utilization_derived"]

for col in continuous_vars:
    if col in df.columns:
        try:
            df[f"{col}_binned"] = pd.qcut(df[col], q=10, duplicates='drop')
        except Exception as e:
            print(f"Could not bin {col}: {e}")
            # If qcut fails (e.g., not enough unique values), use simple binning
            df[f"{col}_binned"] = pd.cut(df[col], bins=10, duplicates='drop')

# WOE & IV Calculation (simplified for demonstration)
# This is a placeholder. A full WOE/IV implementation would involve iterating through bins and calculating WOE/IV.
# For this simulation, we'll just demonstrate the concept by dropping some variables.

# Define a simple function to calculate WOE and IV for a given feature and target
def calculate_woe_iv(df, feature, target):
    event_rate = df[target].mean()
    grouped = df.groupby(feature)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]
    grouped["dist_bad"] = grouped["bad"] / grouped["bad"].sum()
    grouped["dist_good"] = grouped["good"] / grouped["good"].sum()
    grouped["woe"] = np.log(grouped["dist_bad"] / grouped["dist_good"])
    grouped["iv"] = (grouped["dist_bad"] - grouped["dist_good"]) * grouped["woe"]
    iv = grouped["iv"].sum()
    return grouped["woe"], iv

# For the purpose of this simulation, let's create a temporary default_flag for WOE/IV calculation in this section
df["default_flag_temp"] = df["loan_status"].apply(lambda x: 1 if x in ["charged_off", "default"] else 0)

woe_iv_results = {}
for col in df.columns:
    if "_binned" in col or df[col].dtype == "object": # Consider binned features and original categorical features
        # Exclude 'loan_status' itself from WOE calculation as it's the target
        if col not in ['loan_status', 'default_flag_temp']:
            try:
                woe, iv = calculate_woe_iv(df, col, "default_flag_temp")
                woe_iv_results[col] = {"woe": woe, "iv": iv}
            except Exception as e:
                print(f"Could not calculate WOE/IV for {col}: {e}")

# Drop variables with IV < 0.02 (example threshold)
drop_iv_cols = [col for col, res in woe_iv_results.items() if res["iv"] < 0.02]
print(f"Dropping variables with IV < 0.02: {drop_iv_cols}")
df = df.drop(columns=drop_iv_cols, errors='ignore')

# Apply WOE transformation to final features
for col, res in woe_iv_results.items():
    if col not in drop_iv_cols:
        if "_binned" in col:
            # Map WOE values to the binned features
            woe_map = res["woe"].to_dict()
            df[col.replace("_binned", "_woe")] = df[col].map(woe_map)
        elif df[col].dtype == "object":
            # Map WOE values to original categorical features
            woe_map = res["woe"].to_dict()
            df[col.replace("_", "_woe_")] = df[col].map(woe_map)

# Drop the temporary default_flag
df = df.drop(columns=["default_flag_temp"], errors='ignore')

print("Preprocessing and Feature Engineering complete.")
print(df.head())

## Part 3: Target Variables

print("\n--- Part 3: Target Variables ---")

# Target Variables Definition

# PD (Probability of Default)
df["default_flag"] = df["loan_status"].apply(lambda x: 1 if x in ["charged_off", "default"] else 0)

# LGD (Loss Given Default)
# LGD = (funded_amnt - recovered_amount) / funded_amnt
# Handle cases where funded_amnt is zero to avoid division by zero
df["LGD"] = df.apply(lambda row: (row["funded_amnt"] - row["recovered_amount"]) / row["funded_amnt"] if row["funded_amnt"] != 0 else 0, axis=1)
df["LGD"] = df["LGD"].clip(0, 1) # Ensure LGD is between 0 and 1

# EAD (Exposure at Default)
# EAD = funded_amnt - total_rec_prncp
df["EAD"] = df["funded_amnt"] - df["total_rec_prncp"]
df["EAD"] = df["EAD"].clip(lower=0) # Ensure EAD is not negative

print("Target variables defined: default_flag, LGD, EAD.")
print(df[["loan_status", "default_flag", "funded_amnt", "recovered_amount", "total_rec_prncp", "LGD", "EAD"]].head())

## Part 4: Model Building

print("\n--- Part 4: Model Building ---")

### PD Model

# Prepare data for PD model
# Select features that have been WOE transformed or are categorical
# Exclude original continuous variables and target variables

# Identify WOE transformed features
woe_features = [col for col in df.columns if '_woe' in col]

# Identify original categorical features that were not WOE transformed (if any)
categorical_features = [col for col in df.select_dtypes(include='object').columns if col not in ['loan_status', 'default_flag'] and '_woe' not in col]

# Combine all features for the PD model
features_for_pd = woe_features + categorical_features

# Filter out features that might have been dropped due to low IV or other reasons
features_for_pd = [f for f in features_for_pd if f in df.columns]

X = df[features_for_pd].copy() # Create a copy to avoid SettingWithCopyWarning
y = df["default_flag"]

# Impute any remaining NaN or infinite values before get_dummies
for col in X.columns:
    if X[col].dtype in ["int64", "float64"]:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        X[col] = X[col].fillna(X[col].median())
    elif X[col].dtype == "object": # Handle categorical NaNs before get_dummies
        X[col] = X[col].fillna(X[col].mode()[0])

# Now apply get_dummies
X = pd.get_dummies(X, drop_first=True)

# Final check for NaNs and Infs after get_dummies and before split (should be none if previous steps are correct)
# This loop is primarily for debugging and ensuring no new NaNs/Infs were introduced unexpectedly
for col in X.columns:
    if X[col].dtype in ["int64", "float64"]:
        if X[col].isnull().any():
            print(f"Warning: NaN values found in {col} after get_dummies. Imputing with median.")
            X[col] = X[col].fillna(X[col].median())
        if np.isinf(X[col]).any():
            print(f"Warning: Infinite values found in {col} after get_dummies. Replacing with NaN and then imputing with median.")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].median())

print(f"NaN values in X before split: {X.isnull().sum().sum()}")
print(f"Infinite values in X before split: {np.isinf(X).sum().sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"NaN values in X_train after split: {X_train.isnull().sum().sum()}")
print(f"Infinite values in X_train after split: {np.isinf(X_train).sum().sum()}")

# Train Logistic Regression model
pd_model = LogisticRegression(solver='liblinear', random_state=42)
pd_model.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = pd_model.predict_proba(X_test)[:, 1]

# Evaluate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"PD Model AUC: {auc_score:.4f}")

# Calculate KS-statistic
# Sort predicted probabilities and corresponding true labels
df_results = pd.DataFrame({'y_test': y_test, 'y_pred_proba': y_pred_proba})
df_results = df_results.sort_values(by='y_pred_proba', ascending=False)

# Calculate cumulative distributions
df_results['cum_good'] = df_results['y_test'].apply(lambda x: 1 if x == 0 else 0).cumsum() / (y_test == 0).sum()
df_results['cum_bad'] = df_results['y_test'].apply(lambda x: 1 if x == 1 else 0).cumsum() / (y_test == 1).sum()

ks_statistic = np.max(np.abs(df_results['cum_good'] - df_results['cum_bad']))
print(f"PD Model KS-statistic: {ks_statistic:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for PD Model')
plt.legend()
plt.grid()
plt.savefig('roc_curve.png')
plt.close() # Close plot to prevent display issues

# Generate Scorecard (simplified example)
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': pd_model.coef_[0]
})
coefficients['Odds Ratio'] = np.exp(coefficients['Coefficient'])

# Example: Base score and points for each feature
base_score = 500 # Arbitrary base score
points_per_unit = 20 # Arbitrary points per unit of coefficient

# Calculate score for each feature
# This is a simplified approach. A real scorecard involves more complex scaling.
coefficients['Score Points'] = coefficients['Coefficient'] * points_per_unit

print("\nPD Model Scorecard (Simplified):")
print(coefficients)

# Output: Probability of Default + score
df['probability_of_default'] = pd_model.predict_proba(X)[:, 1]
# Simplified score calculation for the entire dataset
df['credit_score'] = base_score + (X.dot(pd_model.coef_[0]) * points_per_unit)

print("\nSample of Probability of Default and Credit Score:")
print(df[['probability_of_default', 'credit_score']].head())

### LGD & EAD Models

# Prepare data for LGD and EAD models
# Use the same features as PD model for consistency, or select relevant ones
features_for_lgd_ead = features_for_pd # Using the same features for simplicity

X_lgd_ead = df[features_for_lgd_ead].copy() # Create a copy to avoid SettingWithCopyWarning

# Handle categorical NaNs before get_dummies
for col in X_lgd_ead.columns:
    if X_lgd_ead[col].dtype == "object":
        X_lgd_ead[col] = X_lgd_ead[col].fillna(X_lgd_ead[col].mode()[0])

X_lgd_ead = pd.get_dummies(X_lgd_ead, drop_first=True) # Ensure all features are numeric

# Final check for NaNs and Infs after get_dummies and before split
for col in X_lgd_ead.columns:
    if X_lgd_ead[col].dtype in ["int64", "float64"]:
        if X_lgd_ead[col].isnull().any():
            print(f"Warning: NaN values found in X_lgd_ead {col} after get_dummies. Imputing with median.")
            X_lgd_ead[col] = X_lgd_ead[col].fillna(X_lgd_ead[col].median())
        if np.isinf(X_lgd_ead[col]).any():
            print(f"Warning: Infinite values found in X_lgd_ead {col} after get_dummies. Replacing with NaN and then imputing with median.")
            X_lgd_ead[col] = X_lgd_ead[col].replace([np.inf, -np.inf], np.nan)
            X_lgd_ead[col] = X_lgd_ead[col].fillna(X_lgd_ead[col].median())

y_lgd = df["LGD"]

X_train_lgd, X_test_lgd, y_train_lgd, y_test_lgd = train_test_split(X_lgd_ead, y_lgd, test_size=0.3, random_state=42)

lgd_model = LinearRegression()
lgd_model.fit(X_train_lgd, y_train_lgd)

y_pred_lgd = lgd_model.predict(X_test_lgd)
y_pred_lgd = np.clip(y_pred_lgd, 0, 1) # Clip LGD predictions between 0 and 1

rmse_lgd = np.sqrt(mean_squared_error(y_test_lgd, y_pred_lgd))
r2_lgd = r2_score(y_test_lgd, y_pred_lgd)

print(f"\nLGD Model RMSE: {rmse_lgd:.4f}")
print(f"LGD Model R²: {r2_lgd:.4f}")

# EAD Model
y_ead = df["EAD"]

X_train_ead, X_test_ead, y_train_ead, y_test_ead = train_test_split(X_lgd_ead, y_ead, test_size=0.3, random_state=42)

ead_model = LinearRegression()
ead_model.fit(X_train_ead, y_train_ead)

y_pred_ead = ead_model.predict(X_test_ead)
y_pred_ead = np.clip(y_pred_ead, 0, None) # Clip EAD predictions to be non-negative

rmse_ead = np.sqrt(mean_squared_error(y_test_ead, y_pred_ead))
r2_ead = r2_score(y_test_ead, y_pred_ead)

print(f"\nEAD Model RMSE: {rmse_ead:.4f}")
print(f"EAD Model R²: {r2_ead:.4f}")

print("Model Building complete.")

## Part 5: Model Monitoring

print("\n--- Part 5: Model Monitoring ---")

### KS Statistic Monitoring

# Simulate KS Statistic over time (for demonstration purposes)
# In a real scenario, you would have new data batches over time

# Let's create a simulated second month data for PSI and KS monitoring
# Sample df_month2 from the fully processed df
df_month2 = df.sample(frac=0.5, random_state=100).copy() # Simulate new data for month 2
# Introduce some drift by slightly altering some features
df_month2["age"] = df_month2["age"] * 1.05 # Simulate age drift
df_month2["annual_income"] = df_month2["annual_income"] * 0.95 # Simulate income drift

# Re-run preprocessing steps for df_month2 to get WOE transformed features
# This is a simplified re-application. In a real system, you'd apply the same transformations as the training data.

# Convert emp_length to numerical
def emp_length_to_int(emp_len):
    if pd.isna(emp_len): return 0
    if isinstance(emp_len, (int, float)): return emp_len # Already converted
    if isinstance(emp_len, str):
        if '10+ years' in emp_len: return 10
        if '< 1 year' in emp_len: return 0
        return int(emp_len.replace(" years", ""))
    return 0 # Default for unexpected types
df_month2["emp_length"] = df_month2["emp_length"].apply(emp_length_to_int)

# Convert loan_term to numerical
# Check if 'loan_term' column exists and is of object type before applying str.replace
if 'loan_term' in df_month2.columns and df_month2['loan_term'].dtype == 'object':
    df_month2["loan_term"] = df_month2["loan_term"].str.replace(" months", "").astype(int)

# Convert earliest_credit_line to datetime and calculate months_since_earliest_credit
# Check if 'earliest_credit_line' column exists before processing
# This block is now removed as df_month2 is sampled from df AFTER earliest_credit_line is dropped from df

# Missing Value Treatment (using median for numeric, mode for categorical)
for col in df_month2.columns:
    if df_month2[col].isnull().sum() > 0:
        if df_month2[col].dtype in ["int64", "float64"]:
            df_month2[col] = df_month2[col].fillna(df_month2[col].median())
        else:
            df_month2[col] = df_month2[col].fillna(df_month2[col].mode()[0])

# Derived Features
df_month2["installment_to_income"] = df_month2["loan_amount"] / df_month2["annual_income"]
df_month2["credit_limit"] = df_month2["revol_bal"] / df_month2["revol_util"]
df_month2["credit_limit"] = df_month2["credit_limit"].replace([np.inf, -np.inf], np.nan).fillna(df_month2["revol_bal"] * 2)
df_month2["credit_utilization_derived"] = df_month2["revol_bal"] / df_month2["credit_limit"]

# Replace infinite values with NaN before binning
for col in df_month2.columns:
    if df_month2[col].dtype in ["float64", "int64"]:
        df_month2[col] = df_month2[col].replace([np.inf, -np.inf], np.nan)

# Feature Binning (fine classing via quantiles for continuous variables)
for col in continuous_vars:
    if col in df_month2.columns:
        try:
            df_month2[f"{col}_binned"] = pd.qcut(df_month2[col], q=10, duplicates='drop')
        except Exception as e:
            df_month2[f"{col}_binned"] = pd.cut(df_month2[col], bins=10, duplicates='drop')

# Apply WOE transformation to df_month2 using WOE values from original df
# This requires re-calculating WOE for df_month2 or storing the WOE maps from original df
# For simplicity, we will re-calculate WOE/IV for df_month2 for demonstration of PSI

df_month2["default_flag_temp"] = df_month2["loan_status"].apply(lambda x: 1 if x in ["charged_off", "default"] else 0)

woe_iv_results_month2 = {}
for col in df_month2.columns:
    if "_binned" in col or df_month2[col].dtype == "object":
        if col not in ['loan_status', 'default_flag_temp']:
            try:
                woe, iv = calculate_woe_iv(df_month2, col, "default_flag_temp")
                woe_iv_results_month2[col] = {"woe": woe, "iv": iv}
            except Exception as e:
                pass

for col, res in woe_iv_results_month2.items():
    if col not in drop_iv_cols:
        if "_binned" in col:
            woe_map = res["woe"].to_dict()
            df_month2[col.replace("_binned", "_woe")] = df_month2[col].map(woe_map)
        elif df_month2[col].dtype == "object":
            woe_map = res["woe"].to_dict()
            df_month2[col.replace("_", "_woe_")] = df_month2[col].map(woe_map)

df_month2 = df_month2.drop(columns=["default_flag_temp"], errors='ignore')

# Predict probabilities for month 2 data
X_month2 = df_month2[features_for_pd].copy() # Create a copy

# Handle categorical NaNs before get_dummies
for col in X_month2.columns:
    if X_month2[col].dtype == "object":
        X_month2[col] = X_month2[col].fillna(X_month2[col].mode()[0])

X_month2 = pd.get_dummies(X_month2, drop_first=True)

# Align columns of X_month2 with X_train (from PD model training)
missing_cols = set(X_train.columns) - set(X_month2.columns)
for c in missing_cols:
    X_month2[c] = 0
# Ensure the order of columns is the same
X_month2 = X_month2[X_train.columns]

# Impute any remaining NaN or infinite values after get_dummies and before prediction
for col in X_month2.columns:
    if X_month2[col].dtype in ["int64", "float64"]:
        if X_month2[col].isnull().any():
            print(f"Warning: NaN values found in X_month2 {col} after get_dummies. Imputing with median.")
            X_month2[col] = X_month2[col].fillna(X_month2[col].median())
        if np.isinf(X_month2[col]).any():
            print(f"Warning: Infinite values found in X_month2 {col} after get_dummies. Replacing with NaN and then imputing with median.")
            X_month2[col] = X_month2[col].replace([np.inf, -np.inf], np.nan)
            X_month2[col] = X_month2[col].fillna(X_month2[col].median())

y_pred_proba_month2 = pd_model.predict_proba(X_month2)[:, 1]

# Calculate KS for month 2
df_results_month2 = pd.DataFrame({'y_test': df_month2["default_flag"], 'y_pred_proba': y_pred_proba_month2})
df_results_month2 = df_results_month2.sort_values(by='y_pred_proba', ascending=False)

df_results_month2['cum_good'] = df_results_month2['y_test'].apply(lambda x: 1 if x == 0 else 0).cumsum() / (df_month2['default_flag'] == 0).sum()
df_results_month2['cum_bad'] = df_results_month2['y_test'].apply(lambda x: 1 if x == 1 else 0).cumsum() / (df_month2['default_flag'] == 1).sum()

ks_statistic_month2 = np.max(np.abs(df_results_month2['cum_good'] - df_results_month2['cum_bad']))

ks_scores = {"Month 1": ks_statistic, "Month 2": ks_statistic_month2}

plt.figure(figsize=(8, 6))
plt.bar(ks_scores.keys(), ks_scores.values(), color=["skyblue", "lightcoral"])
plt.ylabel("KS Statistic")
plt.title("KS Statistic Over Time")
plt.savefig('ks_statistic_over_time.png')
plt.close() # Close plot to prevent display issues

### Population Stability Index (PSI) Monitoring

def calculate_psi(expected, actual, buckettype='quantiles', buckets=10):
    def scale_range(input, min_val, max_val):
        input += -(np.min(input))
        input /= np.max(input) / (max_val - min_val)
        input += min_val
        return input

    breakpoints = np.arange(0, buckets + 1) / buckets

    if buckettype == 'quantiles':
        expected_bins = pd.qcut(expected, q=buckets, labels=False, duplicates='drop')
        actual_bins = pd.qcut(actual, q=buckets, labels=False, duplicates='drop')
    elif buckettype == 'bins':
        min_val = min(min(expected), min(actual))
        max_val = max(max(expected), max(actual))
        breakpoints = scale_range(breakpoints, min_val, max_val)
        expected_bins = pd.cut(expected, bins=breakpoints, labels=False, include_lowest=True)
        actual_bins = pd.cut(actual, bins=breakpoints, labels=False, include_lowest=True)
    else:
        raise ValueError('buckettype can be either \'quantiles\' or \'bins\'')

    expected_counts = np.bincount(expected_bins[~np.isnan(expected_bins)].astype(int), minlength=buckets)
    actual_counts = np.bincount(actual_bins[~np.isnan(actual_bins)].astype(int), minlength=buckets)

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # Replace 0 with a small number to avoid division by zero
    expected_percents[expected_percents == 0] = 0.0001
    actual_percents[actual_percents == 0] = 0.0001

    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi_value

# Calculate PSI for a few key features (e.g., age, annual_income, loan_amount)
psi_values = {}
for feature in ["age", "annual_income", "loan_amount"]:
    if feature in df.columns and feature in df_month2.columns:
        psi = calculate_psi(df[feature], df_month2[feature])
        psi_values[feature] = psi

print("\nPopulation Stability Index (PSI) values:")
for feature, psi in psi_values.items():
    print(f"  {feature}: {psi:.4f}")

plt.figure(figsize=(10, 6))
plt.bar(psi_values.keys(), psi_values.values(), color='lightgreen')
plt.ylabel("PSI Value")
plt.title("Population Stability Index for Key Features (Month 1 vs Month 2)")
plt.axhline(y=0.1, color='orange', linestyle='--', label='Warning (PSI > 0.1)')
plt.axhline(y=0.25, color='red', linestyle='--', label='Alert (PSI > 0.25)')
plt.legend()
plt.savefig('psi_bar_chart.png')
plt.close() # Close plot to prevent display issues

### Data and Feature Drift Visualizations

# Visualize Data and Feature Drift (e.g., using histograms or KDE plots)
selected_features_for_drift = ["age", "annual_income", "revol_util"]

for feature in selected_features_for_drift:
    if feature in df.columns and feature in df_month2.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], color='blue', label='Month 1', kde=True, stat='density', alpha=0.5)
        sns.histplot(df_month2[feature], color='red', label='Month 2', kde=True, stat='density', alpha=0.5)
        plt.title(f"Distribution Drift for {feature}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(f'drift_plot_{feature}.png')
        plt.close() # Close plot to prevent display issues

print("Model Monitoring complete.")

# Save the final processed dataframe to CSV
df.to_csv('credit_risk_processed_data.csv', index=False)
print("Processed data saved to credit_risk_processed_data.csv")

# Generate descriptive statistics and save to a file
descriptive_stats = df.describe(include='all')
descriptive_stats.to_csv('descriptive_statistics.csv')
print("Descriptive statistics saved to descriptive_statistics.csv")

# Generate correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()
print("Correlation heatmap saved to correlation_heatmap.png")

# Generate default rate plots by variable (bin-wise) - Example for 'age'
# This requires re-binning for visualization purposes or using the binned features
# For simplicity, let's use the original 'age' and bin it for visualization

df['age_bins_for_plot'] = pd.cut(df['age'], bins=5)
default_rates_by_age = df.groupby('age_bins_for_plot')['default_flag'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='age_bins_for_plot', y='default_flag', data=default_rates_by_age)
plt.title('Default Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Default Rate')
plt.xticks(rotation=45)
plt.savefig('default_rate_by_age.png')
plt.close()
print("Default rate by age plot saved to default_rate_by_age.png")

# Save scorecard summary to CSV
coefficients.to_csv('scorecard_summary.csv', index=False)
print("Scorecard summary saved to scorecard_summary.csv")

print("All plots and summary files generated.")


