# RiskSense

RiskSense is an end-to-end **credit risk assessment system** that leverages machine learning to predict borrower default risk, explain model decisions, and track experiments using MLOps best practices. The project compares traditional statistical models with advanced tree-based models and focuses on **interpretability, reproducibility, and real-world financial features**.

---

## 🚀 Features

* 📊 **Credit Risk Prediction** using Machine Learning
* ⚖️ **Model Comparison**: Logistic Regression vs XGBoost
* 🔍 **Explainable AI** with SHAP for feature-level insights
* 🧪 **Hyperparameter Optimization** using GridSearchCV
* 📈 **Experiment Tracking & Versioning** with MLflow
* 🛠️ **Feature Engineering** inspired by real-world lending metrics

---

## 🧠 Models Used

* **Logistic Regression**

  * Baseline interpretable model
  * Useful for understanding linear relationships

* **XGBoost Classifier**

  * Handles non-linear interactions
  * Provides superior performance on complex credit data

Models are evaluated and compared based on standard classification metrics.

---

## 🧩 Feature Engineering

Key engineered features include:

* **Payment-to-Income Ratio**
* **Credit Utilization Tiers**
* **Loan Amount vs Income**
* **Loan Term Buckets**
* **Credit Score Bands**

These features help capture borrower behavior and financial stress more effectively than raw variables.

---

## 📊 Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to:

* Explain individual predictions
* Identify global feature importance
* Highlight major risk drivers such as:

  * Credit score
  * Loan term
  * Utilization ratio

This makes RiskSense suitable for **regulated domains like finance**, where transparency is critical.

---

## 🔁 MLOps with MLflow

RiskSense integrates **MLflow** for:

* Experiment tracking
* Logging model parameters and metrics
* Comparing multiple runs
* Model versioning

This ensures reproducibility and clean experimentation workflows.

---

## 🧪 Hyperparameter Tuning

* Implemented using **GridSearchCV**
* Ensures optimal parameter selection
* Prevents overfitting and improves generalization

---

## 🛠️ Tech Stack

* **Programming Language**: Python
* **Machine Learning**: Scikit-learn, XGBoost
* **Explainability**: SHAP
* **MLOps**: MLflow
* **Model Selection**: GridSearchCV
* **Data Handling**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn

---

## 📂 Project Structure (Typical)

```
RiskSense/
│── data/
│── notebooks/
│   └── credit_risk_system.ipynb
│── models/
│── mlruns/
│── README.md
```

---

## 📈 Results & Insights

* XGBoost outperformed Logistic Regression in capturing complex patterns
* Logistic Regression provided strong baseline interpretability
* SHAP analysis aligned well with financial intuition
* MLflow enabled clean comparison across multiple experiments

---

## 🎯 Use Cases

* Bank loan approval systems
* FinTech credit scoring
* Risk analysis dashboards
* Educational ML & MLOps demonstrations

---

## 📌 Future Improvements

* Add real-time inference API (FastAPI/Flask)
* Integrate drift detection
* Expand dataset with temporal credit history
* Deploy models with CI/CD pipelines

---

## 👤 Author

**Prajwal JB**
B.E. Artificial Intelligence & Data Science
BMS College of Engineering, Bengaluru

---

⭐ If you find this project useful, consider starring it!
