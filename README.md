# Hospital Stay Length Prediction

This project is part of an internship assignment focused on predicting the **length of stay** for patients admitted to hospitals using machine learning techniques. The aim is to support hospitals in planning and managing resources efficiently by predicting how long a patient is likely to stay based on admission details, patient demographics, and clinical factors.

---


---

## 🎯 Objective

- Perform exploratory data analysis (EDA) on hospital data.
- Train multiple classification models to predict patient length of stay.
- Compare models based on metrics such as Accuracy, F1-Score, and AUC-ROC.
- Tune hyperparameters using Optuna.
- Select and save the best model for deployment.
- Report on challenges and preprocessing techniques used.

---

## 📊 Dataset Description

The dataset includes the following features:

- Patient demographics (e.g., age, gender)
- Admission and discharge types
- Diagnosis and procedures
- Hospital department and insurance type
- Target variable: Length of Stay (multi-class)

---

## 🧠 Models Implemented

The following models were implemented and evaluated:

- Decision Tree
- Random Forest
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)
- XGBoost
- LightGBM
- CatBoost
- Stacking Ensemble

---

## 📈 Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy
- F1-Score
- AUC-ROC Score

---

## ✅ Best Performing Model

- **Model**: LightGBM Classifier
- **Tuned Parameters**:  
  `n_estimators=100, learning_rate=0.12916, max_depth=6, num_leaves=30, subsample=0.8, colsample_bytree=0.8, class_weight='balanced'`
- **Performance (Test)**:  
  - Accuracy: 0.4168  
  - F1-Score: 0.3873  
  - AUC-ROC: ~0.796
- **Saved as**: `models/best_model.pkl` using `joblib`

---

## 📄 Reports

- [`reports/model_comparison.md`](reports/model_comparison.md): Model performance and recommendation for production.
- [`reports/challenges.md`](reports/challenges.md): Data challenges and strategies used for handling them.

---
