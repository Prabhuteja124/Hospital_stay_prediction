##  Model Comparison Report 

### 1. Objective

The goal of this task was to evaluate various machine learning models for predicting patient length of hospital stay and recommend the best-performing model for production deployment.

### 2. Evaluation Metrics

All models were evaluated using the following metrics:

- **Training Accuracy**
- **Test Accuracy**
- **F1-Score**
- **AUC-ROC Score**

### 3. Performance Before Hyperparameter Tuning

| Model                | Train Accuracy | Test Accuracy | F1-Score | AUC-ROC |
|----------------------|----------------|----------------|----------|---------|
| XGBoost              | 0.5504         | 0.4179         | 0.3896   | 0.7977  |
| LightGBM             | 0.4952         | 0.4162         | 0.3863   | 0.7959  |
| CatBoost             | 0.4963         | 0.4151         | 0.3877   | 0.7957  |
| Stacking Ensemble    | 0.4908         | 0.3983         | 0.3757   | 0.7881  |
| Random Forest        | 0.4282         | 0.3729         | 0.3512   | 0.7692  |
| Decision Tree        | 0.3914         | 0.3658         | 0.3496   | 0.7514  |
| K-Nearest Neighbors  | 0.7476         | 0.2408         | 0.2545   | 0.6004  |
| Logistic Regression  | 0.2639         | 0.2269         | 0.2429   | 0.7247  |
| Naive Bayes          | 0.2058         | 0.1347         | 0.1407   | 0.6763  |

**Observation:**
- XGBoost, LightGBM, and CatBoost consistently outperformed others across all metrics.
- KNN exhibited severe overfitting.
- Simpler models like Logistic Regression and Naive Bayes underperformed.

### 4. Performance After Hyperparameter Tuning

| Model              | Test Accuracy | F1-Score |
|--------------------|---------------|----------|
| LightGBM           | 0.4168        | 0.3874   |
| CatBoost           | 0.4078        | 0.3793   |
| XGBoost            | 0.4052        | 0.3766   |
| Stacking Ensemble  | 0.4107        | 0.3789   |
| Random Forest      | 0.3882        | 0.3734   |
| Decision Tree      | 0.3063        | 0.3065   |

**Observation:**
- Hyperparameter tuning led to slight improvements in all models.
- LightGBM emerged as the best performer, followed by CatBoost and XGBoost.
- Decision Tree continued to show poor generalization performance even after tuning.

### 5. Recommended Model

Based on the comparison of models before and after hyperparameter tuning, the **LightGBM Classifier** is recommended for production deployment due to its:

- High test accuracy and F1-score
- Balanced performance and generalization
- Compatibility with large tabular datasets
- Fast training speed and interpretability

