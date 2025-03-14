# Airlines Delay Prediction

## Project Overview
This project aims to develop a predictive model for airline flight delays. Initially, we attempted modeling using all available features, but the results showed no significant improvement. To address this, we utilized **logistic regression for feature selection** and then applied various **non-linear models** to improve prediction accuracy. Among these, **XGBoost**, which was not introduced in class, showed promising results in enhancing model performance.

## Approach

### 1. Data Collection
- The dataset was sourced from the **Bureau of Transportation Statistics** and **NOAA**, as published on [Kaggle](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations/data).
- It consists of **6 million rows and 24 columns**, including airline information, weather conditions, and flight schedules.

### 2. Feature Selection
- We used **logistic regression** to identify the most relevant features and remove redundant ones.
- A **Variance Inflation Factor (VIF) analysis** and **stepwise selection** helped refine the feature set.

### 3. Model Training
We experimented with several machine learning models, including:
- **Logistic Regression** (baseline)
- **Decision Trees**
- **Random Forest**
- **Gradient Boosting**
- **Generalized Additive Model (GAM)**
- **Neural Networks**
- **K-Nearest Neighbors (KNN)**
- **Extreme Gradient Boosting (XGBoost)**

To handle class imbalance (since only **18.9%** of flights are delayed), we applied **class weighting techniques** and evaluated models based on:
- **Precision**
- **Recall**
- **F1-score** (a priority metric due to class imbalance)

### 4. Model Performance & Results
- **Baseline Logistic Regression** had poor recall, failing to detect delayed flights.
- **Tree-based models** improved performance, but overfitting was a concern.
- **XGBoost with class weighting (4:1)** provided the best balance of recall and precision, making it the most robust choice.

| Model                     | Accuracy | Precision | Recall | F1-score |
|---------------------------|----------|-----------|--------|----------|
| Logistic Regression       | 80.92%   | 0.4314    | 0.009  | 0.0175   |
| Decision Tree (Weighted)  | 65.06%   | 0.2847    | 55.33% | 0.376    |
| Random Forest (Weighted)  | 81.12%   | 0.5238    | 8.5%   | 0.1463   |
| XGBoost (Weighted 4:1)    | **66.09%** | **0.2995** | **58.46%** | **0.3961** |

## Key Findings
- **Departure time** is the strongest predictor of delays.
- **Weather conditions** (precipitation, snow, wind speed) are crucial factors.
- **Class weighting (4:1 ratio)** significantly improves recall for delayed flights.
- **XGBoost outperformed other models**, providing the best trade-off between detecting delays and maintaining precision.

## Future Work
- Hyperparameter tuning for better performance.
- Testing additional ensemble models.
- Exploring deep learning approaches to further improve predictions.

## Repository Information
- **Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations/data)
- **Codebase:** [GitHub Repository](https://github.com/glenyslion/airlines_delay_prediction)

## Contributors
- **Fuqian Zou**
- **Glenys Lion**
- **Iris Lee**
- **Kavya Bhat**
- **Mingze Bian**
