import pandas as pd
import numpy as np
#%%

df = pd.read_csv("data/archive/train.csv")
#%%

df = df.drop("Id", axis=1) #önemsiz bir sütun.
#%%

null_columns = ["PoolQC","Fence","MiscFeature","Alley"]

for col in null_columns:
    df = df.drop(col,axis=1)
# Dropped columns with a high proportion of missing (null) values.
#%%

constant_cols = [col for col in df.columns if df[col].nunique()<=1]
df = df.drop(columns=constant_cols)
# Removed non-informative columns containing only a single unique value or predominantly null values.
#%%

df["MSZoning"] = df["MSZoning"].replace("C (all)","C")
# Performed text corrections and standardization where necessary.
#%%

# Identified columns containing float and string data types with missing values and applied appropriate imputation techniques to handle them.
#%%
num_cols = df.select_dtypes(include="float64").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
#%%
cat_cols = df.select_dtypes(include="str").columns
df[cat_cols] = df[cat_cols].fillna("None")
#%%
## Applied mapping operations to transform categorical values into numerical representations.
qual_map = {
    "Ex":5,
    "Gd":4,
    "TA":3,
    "Fa":2,
    "Po":1,
    "None":0
}
#%%
df["ExterQual"] = df["ExterQual"].map(qual_map)
df["ExterCond"] = df["ExterCond"].map(qual_map)
df["BsmtQual"] = df["BsmtQual"].map(qual_map)
df["BsmtCond"] = df["BsmtCond"].map(qual_map)
df["HeatingQC"] = df["HeatingQC"].map(qual_map)
df["KitchenQual"] = df["KitchenQual"].map(qual_map)
df["FireplaceQu"] = df["FireplaceQu"].map(qual_map)
df["GarageQual"] = df["GarageQual"].map(qual_map)
df["GarageCond"] = df["GarageCond"].map(qual_map)
#%%

df.select_dtypes(include="str").columns
#%%
land_slope_map = {
    "Gtl": 0,
    "Mod": 1,
    "Sev": 2
}
df["LandSlope"] = df["LandSlope"].map(land_slope_map)
#%%

bsmt_exposure_map = {
    "No":0,
    "Mn":1,
    "Av":2,
    "Gd":3,
    "None":0
}
df["BsmtExposure"] = df["BsmtExposure"].map(bsmt_exposure_map)
#%%

bsmt_fin1_map = {
    "Unf":0,
    "LwQ":1,
    "Rec":2,
    "BLQ":3,
    "ALQ":4,
    "GLQ":5,
    "None":0
}
df["BsmtFinType1"] = df["BsmtFinType1"].map(bsmt_fin1_map)
#%%
df["BsmtFinType2"] = df["BsmtFinType2"].map(bsmt_fin1_map)
#%%

functional_map = {
    "Sal":0,
    "Sev":1,
    "Maj2":2,
    "Maj1":3,
    "Mod":4,
    "Min2":5,
    "Min1":6,
    "Typ":7,
    "None":0
}
df["Functional"] = df["Functional"].map(functional_map)
#%%

## Converted selected categorical (string) columns into dummy variables using one-hot encoding.
nominal_cols = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "LotConfig",
    "LandContour",
    "Utilities",
    "LotShape",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "CentralAir",
    "Electrical",
    "GarageType",
    "GarageFinish",
    "PavedDrive",
    "SaleType",
    "SaleCondition"
]
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
#%%

df.shape
#%%

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#%%

y = df["SalePrice"]
x = df.drop("SalePrice", axis=1)
#%%

model = LinearRegression()
model.fit(x, y)
model.score(x,y)
#%%

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=16)
#%%
model.fit(x_train, y_train)
model.score(x_test, y_test)
#%%
y_pred = model.predict(x_test)
#%%

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#%%
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
#%%
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
#%%

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
#%%

modeldict = {
    "Ridge":Ridge(),
    "Lasso":Lasso(),
    "RandomForestRegressor":RandomForestRegressor(random_state=42,n_estimators=200),
    "DecisionTreeRegressor":DecisionTreeRegressor(random_state=42,max_depth=10),
    "ElasticNet":ElasticNet(),
}
#%%

results = []
results.append(["Linear",model.score(x_test, y_test),mae,rmse,r2]) #Lineeri ekledim.
#%%

for name,model in modeldict.items():

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    score = model.score(x_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(name)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    results.append([name,score,mae,rmse,r2])
#%%

df_result = pd.DataFrame(results, columns=["Model","Score","MAE","RMSE","R2"])
#%%

df_result
#%%

import os
#%%
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "archive")
os.makedirs(CLEAN_DIR, exist_ok=True)
csv_path = os.path.join(CLEAN_DIR, "model_results.csv")
df_result.to_csv(csv_path, index=False)