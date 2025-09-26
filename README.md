# Elevate Labs - Task 4: Binary Classification with Logistic Regression

### **Objective**
The goal of this task was to build and evaluate a **Binary Classification Model** using **Logistic Regression** on the provided Breast Cancer Wisconsin dataset. The model's purpose is to predict whether a tumor is Malignant (1) or Benign (0).

### **Workflow Summary**

1.  **Data Preparation**: Irrelevant columns (`id`, `Unnamed: 32`) were dropped. The target variable (`diagnosis`) was converted from categorical (M, B) to numerical (1, 0).
2.  **Feature Scaling**: All 30 feature columns were scaled using `StandardScaler` to ensure the Logistic Regression model performed optimally.
3.  **Data Split**: The data was split into a **80% training set** and a **20% testing set** to assess generalization.
4.  **Model Training**: A `LogisticRegression` model was trained on the scaled training data.
5.  **Evaluation**: The model's performance was rigorously evaluated using key classification metrics.

### **Model Performance Metrics**

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy Score** | [Insert Accuracy Score] | The overall percentage of correct predictions. |
| **Precision (Class 1 - Malignant)** | [Insert Precision for 1] | Of all samples predicted as Malignant, how many were actually Malignant? |
| **Recall (Class 1 - Malignant)** | [Insert Recall for 1] | Of all actual Malignant samples, how many did the model correctly identify? |
| **F1-Score** | [Insert F1-Score] | The harmonic mean of Precision and Recall. |

### **Confusion Matrix**

The Confusion Matrix visually summarizes the model's predictive errors:

| True Label / Predicted Label | Predicted 0 (Benign) | Predicted 1 (Malignant) |
| :--- | :--- | :--- |
| **True 0 (Benign)** | [True Negatives - TN] | [False Positives - FP] |
| **True 1 (Malignant)** | [False Negatives - FN] | [True Positives - TP] |

**Key Finding**: The model demonstrated a [High/Moderate/Low] accuracy, indicating it is [well-suited/moderately suited] for distinguishing between benign and malignant tumors on this test set. The low number of **False Negatives** (Malignant cases missed by the model) is particularly important for this medical application.
