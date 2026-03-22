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

## 👩‍💻 Author

**Akshatha Prabhu**  
Associate Product Manager II (AI/ML) @ HiLabs  
[GitHub](https://github.com/akshathaprabhu22) · [Tableau](https://public.tableau.com/app/profile/akshatha.prabhu6534/vizzes)
