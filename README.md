


#  - Drought Risk Prediction & Classification
## FTL Syria - AI for Climate (Team 8 Capstone Project)

---

## 📌 1. Project Overview & Objectives
* **Challenge:** Problem 2 - Drought Risk Prediction and Classification.
* **Objective:** Develop a robust machine learning classification system to identify weather conditions and predict drought risks accurately using historical meteorological data, supporting early-warning agricultural systems and climate resilience in Syria.
* **Research Question:** Can machine learning identify drought conditions effectively using historical rainfall and temperature patterns?

---

## 🏗️ 2. System Architecture & Data Pipeline

```

[ Historical Climate Data Sources: NASA POWER / CHIRPS ]
↓
[ Data Cleaning, Preprocessing & Monthly Aggregation ]
↓
[ Feature Engineering (Lagged Rainfall, Temperature & Rainfall Anomaly) ]
↓
[ Drought Proxy Target Creation (20th Percentile Historical Baseline) ]
↓
[ Model Training & Comparison: Decision Tree Classifier vs. XGBoost Classifier ]
↓
[ Deep Evaluation & Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix ]

```

---

## 📁 3. Repository Folder Structure
* `data/` : Raw and processed meteorological datasets.
* `notebooks/` : Jupyter / Google Colab notebooks for data exploration and experimentation.
* `src/` : Clean Python scripts, feature engineering functions, and model pipelines.
* `models/` : Saved trained model weights and artifacts.
* `reports/` : Performance evaluations, confusion matrices, and final presentations.
* `README.md` : Comprehensive system architecture and project documentation.

---

## 🚀 4. Sprints & Roadmap (Kanban Board)
* **Sprint 1:** Data Engineering, NASA POWER API Integration, and Drought Proxy Target Creation.
* **Sprint 2:** Model Building (Decision Tree Baseline & XGBoost Classifier).
* **Sprint 3:** Evaluation, Precision/Recall Analysis, and Confusion Matrix interpretation.
* **Sprint 4:** Documentation, GitHub Polish, and Final Presentation Preparation.

```

---
