# Experiment 3 – Regression Analysis using Linear and Regularized Models

This repository contains **Experiment 3** from the *Machine Learning Algorithms Laboratory*.  
The experiment focuses on implementing **Linear Regression**, **Ridge**, **Lasso**, and **Elastic Net** models to predict **loan amount sanctioned**, analyze their performance, and study the **bias–variance tradeoff**.

---

## 📌 Experiment Details

- **Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai  
- **Affiliation:** Anna University  
- **Degree & Branch:** B.E. Computer Science & Engineering  
- **Semester:** VI  
- **Subject Code & Name:** UCS2612 – Machine Learning Algorithms Laboratory  
- **Academic Year:** 2025–2026 (Even Semester)  
- **Batch:** 2023–2027  

---

## 🎯 Objective

To implement and compare:
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Elastic Net Regression  

for **loan amount prediction**, evaluate them using regression metrics, visualize predictions and residuals, and analyze **overfitting, underfitting, bias, and variance**.

---

## 📂 Dataset

- **Loan Amount Prediction Dataset** (Kaggle)  
- **Target Variable:** Loan Amount Request (USD)  

🔗 Dataset link:  
https://www.kaggle.com/datasets/phileinsophos/predict-loan-amount-data

---

## 🧰 Libraries Used

- **Pandas** – Data loading and preprocessing  
- **NumPy** – Numerical computation  
- **Matplotlib** – Plotting and visualization  
- **Seaborn** – Statistical visualization  
- **Scikit-learn** – Model building, preprocessing, evaluation  

---

## 🤖 Regression Models Used

- Linear Regression  
- Ridge Regression (L2 regularization)  
- Lasso Regression (L1 regularization)  
- Elastic Net Regression (L1 + L2 regularization)  

---

## 🧪 Experiment Workflow

### 1️⃣ Data Loading and Exploration
- Load training and testing datasets
- Inspect dataset using `.head()`, `.info()`, `.describe()`
- Visualize feature distributions using box plots

### 2️⃣ Data Preprocessing
- Handle missing values using median imputation
- One-hot encode categorical variables
- Align training and test datasets
- Separate features and target variable

### 3️⃣ Train–Validation Split
- Split data into training and validation sets (80:20)

### 4️⃣ Feature Scaling
- Standardize numerical features using `StandardScaler`

### 5️⃣ Model Training
- Train Linear Regression model
- Tune Ridge, Lasso, and Elastic Net using `GridSearchCV`

### 6️⃣ Model Evaluation
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### 7️⃣ Visualization
- Target variable distribution
- Predicted vs actual values
- Residual plots
- Model comparison bar chart

---

## 📊 Performance Metrics

- **MAE** – Mean Absolute Error  
- **MSE** – Mean Squared Error  
- **RMSE** – Root Mean Squared Error  
- **R² Score** – Coefficient of Determination  

---

## ⚙️ Hyperparameter Tuning Results

| Model | Best Parameters | Best CV R² |
|-----|-----------------|------------|
| Ridge | α = 10 | 0.89 |
| Lasso | α = 0.01 | 0.87 |
| Elastic Net | α = 1, l1_ratio = 0.5 | 0.90 |

---

## 📈 Cross-Validation Performance

| Model | MAE | MSE | RMSE | R² |
|-----|-----|-----|------|----|
| Linear | 24500 | 9.2e8 | 30331 | 0.86 |
| Ridge | 23100 | 8.5e8 | 29155 | 0.88 |
| Lasso | 23800 | 8.9e8 | 29832 | 0.87 |
| Elastic Net | 22000 | 8.1e8 | 28460 | 0.90 |

---

## 🧪 Test Set Performance

| Model | MAE | MSE | RMSE | R² |
|-----|-----|-----|------|----|
| Linear | 25210 | 9.6e8 | 3
