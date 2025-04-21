##  Report on Data Challenges and Solutions :

### 1. Missing Values

**Problem:**  
The dataset contained missing values in several columns.

**Solution:**  
- Used **SimpleImputer** with strategy `most_frequent` for categorical variables and `mean` for numerical variables.
- Justification: Most frequent imputation helps maintain categorical distributions, while mean imputation works well for numeric skewed data.

### 2. Imbalanced Target Variable

**Problem:**  
The target variable (`Length of Stay`) was highly imbalanced, with shorter stays being far more frequent.

**Solution:**  
- Applied **class weights = 'balanced'** in classifiers like LightGBM and RandomForest.
- Also experimented with **SMOTE** for oversampling minority classes during model tuning.


### 3. High Cardinality in Categorical Features

**Problem:**  
Columns like `Department`,`Type_of_Admission` and `Hospital Code` had a large number of unique categories.

**Solution:**  
- Used **Target Encoding** for high cardinality features.
- Used **One-Hot Encoding** for low cardinality features to retain interpretability.



### 4.  Feature Scaling

**Problem:**  
Some models like KNN and Logistic Regression were sensitive to feature scales.

**Solution:**  
- Applied **StandardScaler** to normalize numerical features.
- Tree-based models were excluded from scaling as they are scale-invariant.

### 6.  Model Interpretability

**Problem:**  
Need to explain predictions to stakeholders in the healthcare domain.

**Solution:**  
- Selected **LightGBM** for its interpretability.
- Used **SHAP values** and **feature importance plots** to highlight contributing features.


### 7. Long Training Times

**Problem:**  
Training time was high for ensemble models.

**Solution:**  
- Reduced feature space using feature importance.
- Enabled **multi-threading** with `n_jobs = -1` wherever supported.


#### This structured approach helped in building a reliable and interpretable model for predicting patient hospital stay durations. All challenges were addressed with appropriate preprocessing, feature engineering, and modeling techniques.
