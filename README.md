# House Price Prediction with EmlakJet: ML Technique
#### 
## Project Overview
In this project, various machine learning models were developed to predict house prices using data from EmlakJet. The process involved data preprocessing, feature engineering, and transformation techniques, followed by model training and optimization. Multiple models, including Linear Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, and others, were trained to find the best-performing model. **Optuna** was used for hyperparameter tuning to further improve the best model’s performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Requirements](#requirements)
- [Contact](#contact)

## Data Preprocessing
The data was first preprocessed and cleaned to ensure it was ready for model training:
- **Log10 Transformation**: Applied on target variable (house prices) for initial exploratory analysis and to visualize a linear relationship with features.
- **Box-Cox Transformation**: Applied to normalize skewed features, helping models converge faster and perform better.

## Feature Engineering
Feature engineering is crucial in this project:
- **Feature Creation**: New features such as `price per square meter` were derived from the existing dataset.
- **Feature Transformation**: Skewed features were transformed to reduce the impact of outliers and enhance the model's performance.
  
This step involved:
- Scaling numerical features using **StandardScaler** and **MinMaxScaler**.

## Modeling
Several machine learning models were trained, and their performances were compared:
- **Linear Regression**: Used as a baseline model after applying the **log10 transformation**. Visualization of residuals and model fit was conducted to assess initial performance.
- **RandomForestRegressor**: Ensemble learning method for improved accuracy and robustness.
- **DecisionTreeRegressor**: Simple, interpretable model used to explore feature importance.
- **KNeighborsRegressor**: Non-parametric model for capturing local relationships.
- **BaggingRegressor**: Combines the outputs of multiple models to reduce variance.
- **XGBRegressor**: Gradient boosting algorithm with regularization for preventing overfitting.
- **LGBMRegressor**: LightGBM for high-performance gradient boosting with categorical feature handling.
- **GradientBoostingRegressor**: Powerful boosting model used for complex data structures.
- **MLPRegressor**: Neural network model used to capture non-linear relationships.
- **CatBoostRegressor**: A fast, accurate gradient boosting model that handles categorical features efficiently with minimal tuning.

Each model's performance was evaluated based on:
- **Root Mean Squared Error (RMSE)**

## Hyperparameter Tuning
The best-performing model was further optimized using **Optuna**, a hyperparameter optimization library. Optuna was employed to search for the optimal set of hyperparameters, which further boosted the model's performance.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- catboost
- Matplotlib / Seaborn for visualization
- Optuna (for hyperparameter tuning)

To install the required libraries, you can use:
```bash
pip install -r requirements.txt
```
## contact
For any questions or suggestions, feel free to reach out:
```bash
Email: gokhannergull@gmail.com
```
