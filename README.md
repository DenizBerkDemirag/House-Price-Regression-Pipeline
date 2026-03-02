🏠 House Price Regression Pipeline

End-to-end regression pipeline built on the Kaggle House Prices – Advanced Regression Techniques dataset.
This project focuses on structured data preprocessing, categorical encoding strategies, and comparative evaluation of multiple regression algorithms.

📌 Project Objective

The objective of this project is to:

-Perform systematic data cleaning and preprocessing

-Apply appropriate categorical encoding strategies (ordinal & nominal separation)

-Train and compare multiple regression models

-Evaluate model performance using standard regression metrics

📊 Dataset

-Source: Kaggle

-Dataset: House Prices – Advanced Regression Techniques

-Target Variable: SalePrice

The dataset contains 79 explanatory variables describing residential homes.

🔧 Data Preprocessing Pipeline

The following preprocessing steps were applied:

1️⃣ Data Cleaning

-Dropped non-informative columns (Id)

-Removed columns with excessive missing values

-Removed constant (non-variant) features

2️⃣ Missing Value Handling

-Numerical columns → Median imputation

-Categorical columns → Replaced with "None" where appropriate

3️⃣ Ordinal Encoding (Manual Mapping)

-Applied domain-based ordinal mappings for:

  ExterQual, ExterCond,

  BsmtQual, BsmtCond,

  HeatingQC, KitchenQual,

  FireplaceQu, GarageQual, GarageCond,

  LandSlope,

  BsmtExposure,

  BsmtFinType1, BsmtFinType2,

  Functional

Ordinal relationships were preserved to retain semantic ranking information.

4️⃣ One-Hot Encoding

Nominal categorical variables were transformed using:

  pd.get_dummies(..., drop_first=True)

This prevents multicollinearity in linear models.

🤖 Models Implemented

The following regression algorithms were trained and compared:

  -Linear Regression

  -Ridge Regression

  -Lasso Regression

  -ElasticNet

  -Decision Tree Regressor

  -Random Forest Regressor

📈 Evaluation Metrics

Models were evaluated using:

 -MAE (Mean Absolute Error)

 -RMSE (Root Mean Squared Error)

 -R² Score

All results were exported to: 
model_results.csv

🧠 Key Learning Outcomes

-Proper separation of ordinal and nominal categorical variables

-Manual encoding strategy vs automatic encoding trade-offs

-Impact of multicollinearity in linear regression

-Model bias-variance differences between linear and tree-based methods

-Practical implementation of multi-model benchmarking

🛠 Technologies Used

-Python

-Pandas

-NumPy

-Scikit-learn
