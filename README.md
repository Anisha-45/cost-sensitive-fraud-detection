# 💳 AI-Based Fraud Detection System for Financial Transactions

## 📌 Overview
This project presents a complete machine learning system designed to detect fraudulent credit card transactions. It uses real-world transaction data and applies advanced AI techniques to identify suspicious activities and minimize financial loss.

The system is built as an end-to-end pipeline including data preprocessing, model training, optimization, evaluation, and deployment using a web application.

---

## 🎯 Problem Statement
Traditional rule-based fraud detection systems are unable to adapt to evolving fraud patterns and often generate incorrect predictions.

This project aims to:
- Detect fraudulent transactions accurately
- Reduce financial loss
- Minimize false alerts

---

## 🚀 Key Features
- End-to-end ML pipeline
- Feature engineering (17 features)
- Handling imbalanced data using SMOTE
- Multiple ML models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Hyperparameter tuning using RandomizedSearchCV
- Cost-sensitive threshold optimization
- Interactive Streamlit web application
- Real-time fraud prediction

---

## 🧠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib

---

## 📊 Dataset
- Source: Kaggle Credit Card Fraud Dataset
- Files:
  - `fraudTrain.csv`
  - `fraudTest.csv`
- Total records: ~1.8 million transactions
- Fraud cases: < 0.6% (highly imbalanced dataset)

---

## ⚙️ Project Workflow

1. **Data Preprocessing**
   - Data cleaning
   - Feature engineering
   - Feature scaling (RobustScaler)
   - SMOTE for balancing dataset

2. **Model Training**
   - Logistic Regression (baseline)
   - Random Forest (ensemble model)
   - Gradient Boosting (final best model)

3. **Model Optimization**
   - Hyperparameter tuning using RandomizedSearchCV
   - 5-fold Stratified Cross Validation

4. **Threshold Optimization**
   - Default threshold = 0.50
   - Optimized threshold selected based on financial cost
   - Cost function:
     ```
     Total Cost = (False Positives × $10) + (False Negatives × Fraud Amount)
     ```

5. **Evaluation Metrics**
   - Recall (Fraud Detection Rate)
   - Precision
   - F1 Score
   - AUC-PR (best for imbalanced data)

6. **Deployment**
   - Streamlit web application
   - Real-time fraud prediction interface

---

## 📈 Results

- Best Model: **Gradient Boosting**
- AUC-PR: ~0.82+
- Recall improved significantly after optimization
- Financial loss reduced by ~49.6%

👉 As shown in the project report (Chapter 5), cost-sensitive optimization reduced losses significantly compared to default threshold.

---

## 📊 Visualizations

Generated outputs include:

- EDA Analysis (`eda_overview.png`)
- Model Comparison (`model_comparison.png`)
- ROC Curves (`roc_curves.png`)
- Precision-Recall Curves (`pr_curves.png`)
- Confusion Matrices (`confusion_matrices.png`)
- Feature Importance (`feature_importance.png`)
- Threshold Optimization (`threshold_analysis.png`)
- Before vs After Comparison (`before_after_comparison.png`)

---

## 🖥️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
2️⃣ Add Dataset
fraudTrain.csv
fraudTest.csv
3️⃣ Run ML Pipeline
python main.py
4️⃣ Launch Web App
streamlit run app.py

