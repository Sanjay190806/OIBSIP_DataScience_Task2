# 🇮🇳 Unemployment Rate Analysis in India using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)
![Internship](https://img.shields.io/badge/Oasis%20Infobyte-Data%20Science%20Internship-purple)

---

# 📌 Project Overview

This project was developed as part of the **Oasis Infobyte Data Science Internship (OIBSIP)**.

The objective of this project is to perform a comprehensive analysis of unemployment trends in India and build machine learning models capable of predicting unemployment rates using employment and labor force indicators.

Unemployment is one of the most critical indicators of a country's economic health. The COVID-19 pandemic significantly disrupted labor markets across India, causing sharp fluctuations in employment and unemployment rates. Understanding these patterns can help policymakers, economists, and organizations make informed decisions.

This project combines:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Machine Learning
- Model Evaluation
- Feature Importance Analysis

to generate actionable insights from unemployment data.

---

# 🎯 Project Objectives

The primary objectives of this project are:

### 1. Understand Unemployment Trends

- Analyze unemployment patterns across Indian states.
- Compare urban and rural unemployment.
- Examine monthly and yearly unemployment trends.

### 2. Perform Exploratory Data Analysis

- Discover hidden patterns and relationships.
- Identify regions with the highest and lowest unemployment.
- Understand the impact of employment and labor participation.

### 3. Build Machine Learning Models

Develop predictive models capable of estimating unemployment rates using:

- Employment data
- Labour participation rates
- Regional information
- Temporal information

### 4. Compare Multiple Models

Evaluate and compare:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

using industry-standard metrics.

### 5. Identify Key Factors

Determine the most influential factors affecting unemployment using feature importance analysis.

---

# 📂 Dataset Information

The project uses unemployment datasets containing labor market statistics across India.

## Dataset 1: Unemployment in India

Features:

| Column | Description |
|----------|----------|
| Region | State / Union Territory |
| Date | Observation Date |
| Frequency | Observation Frequency |
| Estimated Unemployment Rate (%) | Target Variable |
| Estimated Employed | Number of Employed Individuals |
| Estimated Labour Participation Rate (%) | Labour Force Participation Rate |
| Area | Urban / Rural |

---

## Dataset 2: Unemployment Rate up to November 2020

Additional dataset containing unemployment information during the COVID-19 period for trend analysis.

---

# 🛠 Technologies Used

## Programming Language

- Python

---

## Data Analysis Libraries

- Pandas
- NumPy

---

## Data Visualization Libraries

- Matplotlib
- Seaborn

---

## Machine Learning Libraries

- Scikit-Learn

Used for:

- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression
- Hyperparameter Tuning
- Cross Validation
- Feature Engineering Pipelines

---

## Model Persistence

- Joblib

---

# 🔄 Project Workflow

## Step 1: Data Collection

- Load unemployment datasets.
- Verify structure and data quality.

---

## Step 2: Data Cleaning

Performed:

✅ Missing Value Analysis

✅ Blank Row Removal

✅ Duplicate Row Removal

✅ Date Format Conversion

✅ Data Validation

---

## Step 3: Feature Engineering

Created new features from the Date column:

| Feature | Description |
|----------|----------|
| Year | Observation Year |
| Month | Observation Month |
| Quarter | Observation Quarter |

After analysis:

- Quarter was removed due to high correlation with Month.
- Frequency was removed because it contained only one unique value.

---

## Step 4: Exploratory Data Analysis (EDA)

Performed detailed analysis of:

### Unemployment Distribution

- Distribution of unemployment rates
- Outlier detection
- Skewness analysis

### State-wise Analysis

- Top 10 states with highest unemployment
- Top 10 states with lowest unemployment

### Urban vs Rural Analysis

- Median comparison
- Distribution comparison
- Outlier analysis

### Correlation Analysis

Studied relationships between:

- Unemployment Rate
- Employment
- Labour Participation Rate
- Year
- Month

---

## Step 5: Outlier Analysis

Applied:

### IQR Method

Calculated:

- Q1
- Q3
- IQR

Detected:

- Extreme unemployment values
- Potential anomalies
- Economic stress periods

---

## Step 6: Data Preprocessing

Implemented a professional Scikit-Learn pipeline.

### Numerical Features

- StandardScaler

### Categorical Features

- One-Hot Encoding

Features encoded:

- Region
- Area

Used:

```python
drop_first=True
```

to avoid the Dummy Variable Trap.

---

## Step 7: Machine Learning Models

### 1. Linear Regression

Used as a baseline model.

Advantages:

- Fast
- Interpretable
- Easy to understand

---

### 2. Random Forest Regressor

Advantages:

- Handles nonlinear relationships
- Robust against overfitting
- Provides feature importance

---

### 3. Gradient Boosting Regressor

Advantages:

- Strong predictive performance
- Learns from previous errors
- Effective on structured datasets

---

# ⚙ Hyperparameter Tuning

To improve model performance:

### Random Forest

Tuned:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf

using:

```python
RandomizedSearchCV
```

---

### Gradient Boosting

Tuned:

- learning_rate
- n_estimators
- max_depth

using:

```python
RandomizedSearchCV
```

---

# 📊 Model Evaluation Metrics

The models were evaluated using:

---

## Mean Absolute Error (MAE)

Measures average prediction error.

Lower is better.

---

## Root Mean Squared Error (RMSE)

Penalizes larger errors.

Lower is better.

---

## R² Score

Measures explained variance.

Range:

```text
1.0 = Perfect Model
0.0 = No Predictive Power
```

Higher is better.

---

## Mean Absolute Percentage Error (MAPE)

Measures percentage prediction error.

Lower is better.

---

# 🔍 Cross Validation

Implemented:

```python
5-Fold Cross Validation
```

Benefits:

- Better reliability
- Reduced overfitting risk
- Improved model validation

Metrics Reported:

- Mean CV R²
- Standard Deviation CV R²

---

# 📈 Visualizations Included

### Distribution of Unemployment Rate

Understanding target variable behavior.

---

### Top 10 States by Unemployment

Identify high-risk regions.

---

### Top 10 States with Lowest Unemployment

Identify successful labor markets.

---

### Urban vs Rural Boxplot

Compare unemployment distributions.

---

### Correlation Heatmap

Study feature relationships.

---

### Employment Distribution

Analyze employment patterns.

---

### Labour Participation Distribution

Analyze workforce participation.

---

### Year-wise Trend Analysis

Study long-term unemployment trends.

---

### Month-wise Trend Analysis

Identify seasonal patterns.

---

### Feature Importance Analysis

Identify most influential predictors.

---

### Actual vs Predicted Plot

Evaluate model predictions visually.

---

### Residual Analysis

Check prediction quality.

---

# 🔑 Key Findings

### 1. Significant Regional Differences Exist

Certain states consistently exhibit higher unemployment rates than others.

---

### 2. Urban Areas Show Greater Variability

Urban unemployment displays greater fluctuations and more extreme outliers.

---

### 3. Employment Negatively Correlates with Unemployment

Higher employment levels are generally associated with lower unemployment rates.

---

### 4. Labour Participation Influences Unemployment

Participation rates contribute significantly to labor market dynamics.

---

### 5. Seasonal Patterns Are Present

Monthly analysis indicates recurring fluctuations in unemployment levels.

---

### 6. Machine Learning Can Effectively Predict Unemployment

Tree-based models outperform the baseline linear regression model.

---

# 🏆 Best Model

The best-performing model was selected using:

- Lowest RMSE
- Highest R² Score
- Cross Validation Stability

The final model was saved using:

```python
joblib.dump()
```

allowing future reuse without retraining.

---

# 📁 Project Structure

```text
OIBSIP_DataScience_Task2/
│
├── Unemployment_Analysis.ipynb
├── Unemployment in India.csv
├── Unemployment_Rate_upto_11_2020.csv
├── README.md
├── best_unemployment_model.joblib
└── Screenshots/
```

---

# 🚀 Future Improvements

Potential enhancements:

### Advanced Models

- XGBoost
- LightGBM
- CatBoost

---

### Time Series Forecasting

- ARIMA
- SARIMA
- Prophet
- LSTM

---

### Dashboard Development

- Streamlit
- Power BI
- Tableau

---

### Additional Features

- GDP Growth
- Inflation Rate
- Industrial Production
- Population Statistics

---

### Real-Time Monitoring

- Automated data pipelines
- Model monitoring
- Data drift detection

---

# 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Data Cleaning
- Data Validation
- Feature Engineering
- Exploratory Data Analysis
- Data Visualization
- Machine Learning Pipelines
- Regression Modeling
- Hyperparameter Tuning
- Cross Validation
- Feature Importance Analysis
- Model Evaluation
- Model Deployment Preparation

---

# 👨‍💻 Author

### Sanjay K

Data Science Intern – Oasis Infobyte

Aspiring Machine Learning Engineer

GitHub:
https://github.com/Sanjay190806

LinkedIn:
(Add your LinkedIn profile link)

---

# ⭐ Conclusion

This project demonstrates a complete end-to-end Machine Learning workflow for unemployment analysis in India. By combining data cleaning, exploratory analysis, feature engineering, predictive modeling, and model evaluation, the project provides valuable insights into labor market trends and establishes a strong foundation for future economic forecasting applications.

The developed solution showcases practical Data Science and Machine Learning skills applicable to real-world business and policy decision-making scenarios.