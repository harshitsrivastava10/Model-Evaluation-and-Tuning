# Machine Learning Model Evaluation & Hyperparameter Tuning

## Project Overview
This repository contains Week 3 of the AI/ML Internship tasks. Building upon the exploratory data analysis from Week 2, this project focuses on implementing, evaluating, and optimizing machine learning models to predict community fragrance ratings based on dataset metrics.

## Methodology
1. **Data Preprocessing:** The cleaned Fragrantica dataset was imported, and high-cardinality textual features were dropped to prevent dimensionality explosion. Categorical variables (Gender demographics) were processed using One-Hot Encoding (`pd.get_dummies`).
2. **Baseline Modeling:** Two models were trained on an 80/20 train-test split:
   * **Ridge Regression:** To establish a linear baseline.
   * **Random Forest Regressor:** To capture non-linear relationships and interactions between features.
3. **Evaluation Metrics:** Models were assessed using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the $R^2$ Score to determine variance predictability.
4. **Hyperparameter Tuning:** `GridSearchCV` was deployed on the Random Forest model to test combinations of `n_estimators`, `max_depth`, and `min_samples_split` using 3-fold cross-validation.

## Repository Contents
* `model_evaluation_and_tuning.py`: The core pipeline script for model training and visualization generation.
* `docs/performance_report.md`: A detailed breakdown of model performance, metrics explanation, and final conclusions.
