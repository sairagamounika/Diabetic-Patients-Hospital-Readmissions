# 🩺 Predicting Hospital Readmissions for Diabetic Patients Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Libraries](https://img.shields.io/badge/Libraries-pandas%2C%20scikit--learn%2C%20xgboost%2C%20matplotlib%2C%20seaborn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20%2F%20Kaggle-green)

---

## 📌 Problem Statement
Unplanned hospital readmissions significantly increase healthcare costs and strain hospital resources.  
For diabetic patients—who often require continuous monitoring and medication adjustments—these readmissions are frequent yet preventable.

This project aims to **predict whether a diabetic patient will be readmitted within 30 days after discharge** using demographic, clinical, and hospital-stay data.  
By identifying high-risk patients early, hospitals can intervene proactively to improve outcomes and reduce costs.

---

## 🧠 Dataset Overview
| Attribute | Details |
|:-----------|:---------|
| **Source** | Kaggle / UCI ML Repository – *Diabetes 130-US Hospitals (1999–2008)* |
| **Records** | 100,000+ patient encounters from 130 hospitals |
| **Features** | 50 attributes (13 numerical, 37 categorical) |
| **Target Variable** | `readmitted` → converted to binary (`1` = <30 days, `0` = not readmitted) |
| **Challenge** | Highly imbalanced data (~11 % positive class) |

---

## ⚙️ Methodology

### 🧹 Data Cleaning & Preprocessing
- Dropped high-null columns: `weight`, `medical_specialty`, `payer_code`
- Imputed missing values in `race`, `max_glu_serum`, and `A1Cresult`
- Removed invalid gender and expired/hospice discharges
- Encoded variables:
  - **Binary:** `gender`, `change`, `diabetesMed`
  - **Label Encoding:** `admission_source_id`, `discharge_disposition_id`
  - **One-Hot Encoding:** `race`, `A1Cresult`, `max_glu_serum`
- Applied **log transformation** to skewed variables (`num_medications`, `num_visits`)

---

### 🧩 Feature Engineering
- **ICD-9 Code Categorization:** Mapped 1,000 + diagnosis codes in `diag_1–3` into 9 broader disease groups  
  *(Circulatory, Respiratory, Digestive, Diabetes, Injury, etc.)*
- **Aggregated Visits:** Combined inpatient, outpatient, and emergency visits → `num_visits`
- **Log Features:** `num_medications_log`, `num_visits_log`
- **Feature Selection:** Recursive Feature Elimination (RFE) with XGBoost  
  → Kept 18 key predictors (age, admission type, stay length, diagnoses, A1C, med changes, visits)

---

### ⚖️ Handling Class Imbalance
Techniques explored:
- **SMOTE** – Synthetic Minority Oversampling  
- **Balanced Class Weights** – in Logistic Regression / Tree models  
- **Balanced Bagging** – for ensemble robustness  

> SMOTE improved recall but slightly reduced precision — a common trade-off when optimizing for sensitivity in healthcare prediction.

---

### 🤖 Model Development
| Model | Techniques | Recall | Precision | Accuracy | AUC-ROC |
|:------|:------------|:------:|:----------:|:---------:|:-------:|
| Logistic Regression | Baseline + balanced weights | 0.56 | 0.17 | 0.63 | 0.64 |
| Decision Tree | Tuned (`max_depth=5`) | 0.64 | 0.16 | 0.57 | 0.64 |
| Random Forest | SMOTE + Tuning | 0.55 | 0.18 | 0.66 | 0.65 |
| XGBoost | Tuned (`300 trees`, `lr=0.01`) | 0.42 | 0.20 | 0.74 | 0.66 |
| Balanced Bagging | Bootstrap Aggregation | 0.58 | 0.18 | 0.65 | 0.66 |
| **Voting Ensemble (DT + RF + XGB)** | Final Model | **0.43** | **0.20** | **0.73** | **0.66** |

> 🏆 **Best Individual Model:** XGBoost  
> 🧩 **Best Overall Model:** Ensemble (Voting Classifier) — stable and generalizable performance.

---

## 📊 Results & Insights
- **Recall prioritized** to minimize false negatives (missing high-risk patients).  
- **Top Predictors:**
  - ICD-9 disease category  
  - Number of medications  
  - Time in hospital (length of stay)  
  - Number of visits (`num_visits`)  
  - A1C test results + medication changes  
  - Discharge disposition  

**Key Findings**
- Patients with **longer hospital stays**, **frequent visits**, and **medication adjustments** are more likely to be readmitted within 30 days.  
- Ensemble models captured non-linear feature interactions better than linear baselines.  

---

## 🏥 Discussion & Business Impact
- Predictive modeling enables hospitals to **flag high-risk diabetic patients before discharge**.  
- Integrating these models in clinical workflows can:
  - Improve patient follow-ups and discharge planning  
  - Reduce readmission rates and resource strain  
  - Support data-driven healthcare decisions  
- The project aligns with literature (Shang et al., 2021) showing that **tree-based and ensemble models** outperform linear approaches for complex medical data.

---

## 🚀 Conclusion
- The **Voting Ensemble (RF + XGB + DT)** achieved **AUC ≈ 0.66** and **Recall ≈ 0.43**, providing a balanced trade-off between accuracy and sensitivity.  
- Predictive models like this can assist—**not replace**—clinical judgment in readmission risk analysis.  
- This work demonstrates how interpretable ML solutions can support better diabetic-patient outcomes and hospital efficiency.

---

## 🔮 Future Work
- Add individual medication features (insulin, metformin)  
- Apply **PCA** for dimensionality reduction  
- Explore **SVM**, **KNN**, and **Deep Learning (LSTM / CNN)**  
- Deploy a **Streamlit / Flask app** for interactive prediction  
- Evaluate **model fairness** across patient demographics

---

## 🧰 Tech Stack
**Languages:** Python  
**Libraries:** pandas · numpy · scikit-learn · xgboost · imbalanced-learn · matplotlib · seaborn  
**Techniques:** SMOTE · RFE · GridSearchCV · Stratified K-Fold Cross Validation  
**Tools:** Jupyter Notebook · GitHub · Kaggle  

---

## 📚 References
1. Alturki et al. (2019) – Predictors of Readmissions and Length of Stay for Diabetes-Related Patients  
2. Rubin (2015) – Hospital Readmission of Patients with Diabetes  
3. Shang et al. (2021) – The 30-Day Hospital Readmission Risk in Diabetic Patients  
4. Rodriguez-Gutierrez et al. (2019) – Racial and Ethnic Differences in 30-Day Readmissions  
5. Hammoudeh et al. (2018) – Predicting Hospital Readmission Among Diabetics Using Deep Learning  
6. Hasan et al. (2010) – Hospital Readmission in General Medicine Patients: A Prediction Model  


