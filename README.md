# 🩺 Diabetes Prediction Model

A machine learning project that predicts the likelihood of diabetes in patients 
based on behavioral and health indicator data collected by the CDC's Behavioral 
Risk Factor Surveillance System (BRFSS) 2015. Built using Python with a focus 
on exploratory data analysis, feature selection, and binary classification modeling.

---

## 📌 Objective

To develop a reliable classification model that predicts whether a patient is 
diabetic or pre-diabetic based on 21 health and lifestyle indicators — enabling 
early detection and supporting preventive healthcare decisions.

> Diabetes affects over **38.4 million individuals** annually in the U.S. and is 
> the **8th leading cause of death**. Prediction models help identify at-risk 
> individuals and assist healthcare systems in planning drug demand, therapies, 
> and patient care.

---

## 📂 Dataset

- **Source:** [Diabetes Health Indicators Dataset — Kaggle (Alex Teboul)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Origin:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015
- **Records:** 70,692 survey responses (balanced 50-50 split)
- **Features:** 21 health indicator variables + 1 binary target variable
- **Target Variable:** `Diabetes_binary` — 0 = No Diabetes, 1 = Prediabetes or Diabetes
- **Missing Values:** None — dataset is pre-cleaned

| Feature | Description |
|---|---|
| HighBP | High blood pressure indicator |
| HighChol | High cholesterol indicator |
| BMI | Body Mass Index (only continuous variable) |
| Smoker | Smoking status |
| HeartDiseaseorAttack | History of heart disease |
| PhysActivity | Physical activity in past 30 days |
| HvyAlcoholConsump | Heavy alcohol consumption |
| GenHlth | General health self-rating (1–5) |
| Age | Age group category |
| Income | Income level |
| + 11 more indicators | Lifestyle & demographic variables |

---

## 🔧 Tech Stack

<img src="https://upload.wikimedia.org/wikipedia/commons/f/f8/Python_logo_and_wordmark.svg" alt="Python" width="90"/>

| Library | Usage |
|---|---|
| Pandas | Data loading & manipulation |
| NumPy | Numerical computations |
| Matplotlib / Seaborn | EDA & visualization |
| Scikit-learn | Model building & evaluation |
| Statsmodels | OLS Linear Regression |

---

## 🔍 Project Workflow

1. **Data Loading & Exploration** — Shape, dtypes, null checks, descriptive statistics
2. **Exploratory Data Analysis (EDA)** — Demographic trends (age, income, education, sex), BMI analysis, correlation heatmap
3. **Feature Selection** — Dropped multicollinear and weakly correlated features (`CholCheck`, `Stroke`, `Fruits`, `Sex`, `Veggies`, `AnyHealthcare`, `NoDocbcCost`, `MentHlth`, `PhysHlth`, `DiffWalk`, `Education`)
4. **Model Building** — Three classifiers compared: Linear Regression, KNN, Logistic Regression
5. **Model Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC

---

## 📊 Model Results

| Model | Accuracy | Notes |
|---|---|---|
| Linear Regression (OLS) | R² = 0.309 | Poor fit — binary target not suited for OLS |
| K-Nearest Neighbours (k=2) | 64.3% | Better fit but low interpretability |
| **Logistic Regression** | **74.2%** | **Best overall — selected model** |

### Logistic Regression — Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 — No Diabetes | 0.75 | 0.72 | 0.74 |
| 1 — Diabetic / Pre-diabetic | 0.74 | 0.76 | 0.75 |
| **Overall Accuracy** | | | **74.2%** |

- **ROC-AUC Score: 0.82** — strong ability to distinguish between diabetic and non-diabetic cases
- **True Negative Rate:** Logistic Regression (38.31%) outperformed KNN (23.84%) — critical for avoiding false diagnoses

---

## 📈 Key Findings

- **Top predictors of diabetes:** General Health (`GenHlth`), High Blood Pressure (`HighBP`), BMI, High Cholesterol (`HighChol`), and Age
- **Demographic trends:**
  - Diabetes prevalence increases significantly with age (60+ age groups most at risk)
  - Lower income groups show higher diabetes rates — likely linked to healthcare access and diet
  - Sex was not a significant predictor — near 50-50 split across both classes
- **Lifestyle factors matter:** Heavy alcohol consumption, physical inactivity, and high cholesterol are strongly associated with diabetes onset
- **Logistic Regression selected** as the final model — better true negative rate, higher interpretability, and more coherent coefficients vs KNN

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/akshathaprabhu22/Diabetes-Prediction-Model.git

# Navigate to the project
cd Diabetes-Prediction-Model

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

# Open the notebook
jupyter notebook "Diabetes Prediction Model.ipynb"
```

---

## 🔄 Future Improvements

### 1. Model Upgrades — Go Beyond Logistic Regression

The current best model sits at 74.2%. Adding ensemble models is expected to push this to 80–85% based on published benchmarks on this dataset.

```python
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# XGBoost — best performer for mixed feature types
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
xgb.fit(X_train, y_train)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb.predict(X_test)))

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
```

---

### 2. Fix Class Imbalance with SMOTE

The original BRFSS dataset is 86.1% non-diabetic. Even the balanced split used here could benefit from synthetic oversampling to improve recall on the diabetic class.

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

logreg.fit(X_resampled, y_resampled)
```

---

### 3. Hyperparameter Tuning

The current KNN uses `k=2` (likely overfitting) and Logistic Regression throws a `ConvergenceWarning` due to default `max_iter`. Both need tuning.

```python
from sklearn.model_selection import GridSearchCV

# Find optimal k for KNN
knn_params = {'n_neighbors': [3, 5, 7, 10, 15, 20]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='f1')
knn_grid.fit(X_train, y_train)
print("Best k:", knn_grid.best_params_)

# Fix Logistic Regression convergence warning
logreg = LogisticRegression(max_iter=1000, solver='saga')
logreg.fit(X_train, y_train)
```

---

### 4. Replace Manual Feature Selection with RFE

Features are currently dropped manually based on correlation thresholds. Recursive Feature Elimination (RFE) provides a more principled, automated approach.

```python
from sklearn.feature_selection import RFE

rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]
print("RFE Selected Features:", list(selected_features))
```

---

### 5. Add Cross-Validation

Results are currently based on a single 70/30 train-test split, which can vary depending on the random state. 10-fold cross-validation provides a more reliable accuracy estimate.

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
print(f"10-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

---

### 6. Production-Ready Data Ingestion

The dataset is currently loaded from a static local CSV. Here's how to make ingestion automated and scalable.

```python
# Option 1: Load directly from Kaggle API
import kaggle
kaggle.api.dataset_download_files(
    'alexteboul/diabetes-health-indicators-dataset',
    path='./data', unzip=True
)

# Option 2: Load from AWS S3 staging bucket
import boto3
import pandas as pd

s3 = boto3.client('s3')
s3.download_file('your-bucket', 'diabetes_data.csv', '/tmp/diabetes_data.csv')
df = pd.read_csv('/tmp/diabetes_data.csv')
```

**Recommended pipeline architecture:**
```
CDC BRFSS Source → Kaggle API → AWS S3 (Raw) → ETL Script → Model Training
```

---

### 7. Model Deployment

The project currently ends at evaluation. Save the trained model so it can be deployed and used for inference on new patient data.

```python
import joblib

# Save trained model
joblib.dump(logreg, 'diabetes_model.pkl')

# Load and predict on new patient data
model = joblib.load('diabetes_model.pkl')
new_patient = [[1, 1, 28.5, 0, 0, 1, 0, 3, 8, 6]]  # example feature values
prediction = model.predict(new_patient)
print("Diabetes Risk:", "Yes" if prediction[0] == 1 else "No")
```

---

### 📋 Improvement Summary

| Improvement | Effort | Expected Impact |
|---|---|---|
| Add XGBoost / Random Forest | Low | +5–10% accuracy |
| Fix convergence warning (`max_iter`) | Very Low | Cleaner, stable results |
| Tune KNN `k` value via GridSearchCV | Low | Better KNN baseline |
| Add 10-fold cross-validation | Low | More reliable accuracy estimate |
| Add SMOTE for class imbalance | Medium | Better recall on diabetic class |
| Recursive Feature Elimination (RFE) | Medium | More principled feature selection |
| Kaggle API / S3 data ingestion | Medium | Automated, scalable pipeline |
| Save model with joblib | Very Low | Deployable artifact |

---

## 👩‍💻 Author

**Akshatha Prabhu**  
Associate Product Manager II (AI/ML) @ HiLabs  
[GitHub](https://github.com/akshathaprabhu22) · [Tableau](https://public.tableau.com/app/profile/akshatha.prabhu6534/vizzes)
