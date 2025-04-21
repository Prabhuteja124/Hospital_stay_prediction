import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
def Null_values(data):
    missing_values = data.isnull().sum()
    print("Missing Values in Each Column: \n", missing_values)

def detect_outliers_iqr(data, continuous_columns):
    """
    Detects outliers using IQR method for continuous columns.
    
    Parameters:
    - data: pd.DataFrame
    - continuous_columns: list of continuous numerical column names
    
    Returns:
    - pd.DataFrame with outlier count and percentage for each column
    """
    outlier_summary = []

    for col in continuous_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_count = outliers.shape[0]
        outlier_percentage = (outlier_count / len(data)) * 100

        outlier_summary.append({
            'Column': col,
            'Outlier Count': outlier_count,
            'Outlier Percentage': round(outlier_percentage, 2)
        })

    outlier_df = pd.DataFrame(outlier_summary)
    print("📊 Outlier Summary (IQR Method):\n")
    # print(outlier_df)
    return outlier_df

import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplots(data, columns, n_cols=3, figsize=(20, 20), color_palette='Set2'):
    """
    Plots boxplots for the given numerical columns to check for outliers.

    Parameters:
    - data: pd.DataFrame
    - columns: list of column names (numerical)
    - n_cols: number of plots per row (default=3)
    - figsize: size of the entire figure (default=(20,20))
    - color_palette: seaborn color palette (default='Set2')
    """

    sns.set(style="whitegrid")
    n_rows = (len(columns) + n_cols - 1) // n_cols

    plt.figure(figsize=figsize, facecolor='white')
    plt.suptitle("📦 Boxplots for Outlier Detection", fontsize=24, fontweight='bold', color='black', y=1.02)

    for idx, column in enumerate(columns, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        sns.boxplot(y=data[column], palette=color_palette, ax=ax)
        ax.set_title(f'{column}', fontsize=16)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('')
        ax.grid(True)

    plt.tight_layout(pad=2.0)
    plt.show()



def evaluate_model(name, model, x_resampled, y_resampled, x_test, y_test):
    print(f"\n Training model: {name}")
    model.fit(x_resampled, y_resampled)
    y_train_pred = model.predict(x_resampled)
    y_test_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        try:
            y_test_proba = model.predict_proba(x_test)
            if y_test_proba.shape[1] > 2:
                auc_score = roc_auc_score(y_test, y_test_proba, multi_class='ovr')
            else:
                auc_score = roc_auc_score(y_test, y_test_proba[:, 1])
        except:
            auc_score = None
    else:
        auc_score = None

    train_acc = accuracy_score(y_resampled, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    print(f" Train Accuracy: {train_acc:.4f}")
    print(f" Test Accuracy: {test_acc:.4f}")
    print(f" F1-score: {f1:.4f}")
    if auc_score is not None:
        print(f" AUC-ROC: {auc_score:.4f}")
    print(" Classification Report:\n", classification_report(y_test, y_test_pred))

    return {
        "Model": name,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "F1-score": f1,
        "AUC-ROC": auc_score
    }
