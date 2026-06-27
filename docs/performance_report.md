# Model Performance & Tuning Report

## 1. Metric Selection Rationale
To evaluate the regression models predicting community ratings (a continuous variable from 1.0 to 5.0), the following metrics were selected:
* **MAE (Mean Absolute Error):** Chosen for its high interpretability. It tells us exactly how far off, on average, our rating predictions are from the actual user scores.
* **RMSE (Root Mean Squared Error):** Chosen to penalize larger errors. If a model predicts a 4.5 for a fragrance that is actually a 2.0, RMSE will flag this aggressive variance more prominently than MAE.
* **$R^2$ Score (Coefficient of Determination):** Chosen to measure how much of the variance in the target variable (Rating) is explained by our input features (Votes, Gender).

## 2. Baseline Model Performance
* **Ridge Regression:** Showed limited capability in predicting ratings, indicating that the relationship between popularity (votes), gender marketing, and final rating is not strictly linear.
* **Random Forest Regressor:** The ensemble method established a stronger baseline, reducing both MAE and RMSE, proving better equipped to handle the variance in community sentiment.

## 3. Hyperparameter Tuning Results
Using `GridSearchCV`, the Random Forest model was optimized. The grid search iterated through tree counts (50, 100, 200) and depth constraints to prevent overfitting on the training data.
* **Best Parameters:** The grid search identified the optimal combination (typically deeper trees with a higher `n_estimators` count for this specific dataset).
* **Impact:** The tuned model showed a marginal improvement in stability (RMSE reduction) on unseen data compared to the default baseline parameters.

## 4. Final Observations & Feature Analysis
As seen in the generated `feature_importance.png`, the volume of **Votes** carries the highest predictive weight in the model. However, the relatively low overall $R^2$ score across all models indicates a fundamental truth about subjective datasets: *demographics and popularity alone cannot accurately predict scent preference.* To significantly improve this model in the future, Natural Language Processing (NLP) embeddings of the specific scent notes (e.g., Bergamot, Oud, Vanilla) would need to be engineered into the feature space.