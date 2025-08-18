# Bank Fraud Detection

## Description
This project explores bank fraud detection using advanced machine learning techniques. The goal is to build a robust model capable of detecting fraudulent transactions with high accuracy.

---

## Dataset
The dataset used is **`creditcard.csv`**, which contains bank transactions made with credit cards. It is highly imbalanced, with a small proportion of fraudulent transactions (0.17% of the data). Each transaction is described by 30 features (anonymized for confidentiality reasons) and a binary target variable (`Class`) indicating whether the transaction is fraudulent (1) or not (0).

---

## Project Steps

### 1. **Data Visualization**
- Exploratory analysis to understand class distribution.
- Visualization of distributions and correlations between features.

### 2. **Exploration of Basic Models**
- Testing classical machine learning models (Logistic Regression, Random Forest, SVM, etc.).
- Using the SMOTE method to handle class imbalance.

### 3. **Optimization of XGBClassifier (scikit-learn)**
- Using `XGBClassifier` with hyperparameter search (Optuna).
- Optimization to maximize accuracy and ROC AUC.

### 4. **Ensemble Learning (Stacking)**
- **First layer**: Using basic models (ExtraTrees, Gaussian NB, MLP, DecisionTree, XGB, LGB, CAT, HistGB, Logistic Regression, Random Forest).
- **Second layer**: A simple neural network (`MLPClassifier`) to combine the predictions of the first-layer models.
- Goal: Improve the robustness and accuracy of the final model.

### 5. **Optimization with XGBoost (Original Library)**
- Switching to the original `xgboost` library for finer optimization.
- Tuning hyperparameters to maximize performance.

---

## Results
The final optimized model achieved:
- **Accuracy**: 96.7%  
- **ROC AUC**: 99%  
- **PR AUC**: 88%  

---

## Project Structure
- `creditcard.csv`: Dataset of bank transactions.  
- `creditcard_fraud.ipynb`: Jupyter Notebook containing the full analysis, from visualization to model optimization.  
