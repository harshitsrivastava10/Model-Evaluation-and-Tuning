import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create images folder for outputs
os.makedirs('images', exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Load the cleaned data from Week 2
# (Assuming 'cleaned_dataset.csv' is in the same directory)
try:
    df = pd.read_csv('cleaned_dataset.csv')
except FileNotFoundError:
    print("Please ensure 'cleaned_dataset.csv' is uploaded to the directory.")
    exit()

# 2. Feature Selection & Preprocessing
# We will drop high-cardinality text columns (like specific notes) to prevent memory crashes
# and focus on Votes and Gender to predict the Rating.
features_to_keep = ['Rating', 'Votes', 'Gender']
df_model = df[[col for col in features_to_keep if col in df.columns]].dropna()

# One-Hot Encode categorical variables (e.g., Gender)
df_encoded = pd.get_dummies(df_model, drop_first=True)

# Define Features (X) and Target (y)
X = df_encoded.drop('Rating', axis=1)
y = df_encoded['Rating']

# Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Baseline Models
print("--- Training Baseline Models ---")

# Model A: Ridge Regression (Linear)
lr_model = Ridge(alpha=1.0)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Model B: Random Forest (Ensemble)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 4. Evaluation Function
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} Performance:\n MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}\n")
    return mae, rmse, r2

evaluate_model("Baseline Ridge Regression", y_test, lr_preds)
evaluate_model("Baseline Random Forest", y_test, rf_preds)

# 5. Hyperparameter Tuning (GridSearchCV on Random Forest)
print("--- Commencing Hyperparameter Tuning ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                           param_grid=param_grid, 
                           cv=3, 
                           scoring='neg_mean_squared_error', 
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print(f"Best Parameters Found: {grid_search.best_params_}")
tuned_preds = best_rf.predict(X_test)
evaluate_model("Tuned Random Forest", y_test, tuned_preds)

# 6. Visualizations
# Chart 1: Predicted vs Actual (Tuned Model)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=tuned_preds, alpha=0.6, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Perfect prediction line
plt.title('Actual vs. Predicted Fragrance Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.savefig('images/predicted_vs_actual.png', dpi=300)

# Chart 2: Feature Importance
plt.figure(figsize=(8, 5))
feature_importances = best_rf.feature_importances_
sns.barplot(x=feature_importances, y=X.columns, palette='viridis')
plt.title('Feature Importance in Tuned Random Forest')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.savefig('images/feature_importance.png', dpi=300)

print("\nProcess Complete. Visualizations saved to 'images/' folder.")