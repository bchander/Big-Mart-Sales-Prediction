# -*- coding: utf-8 -*-
"""
BigMart Sales Prediction!

Problem Statement
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. 
The aim is to build a predictive model and predict the sales of each product at a particular outlet.
Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales.
Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly. 

Created on Aug 20th 2025
Modified on 

@created by: Bhanu Chander V 
"""
# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error as mse, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
#matplotlib.use("Agg")
import seaborn as sns
from tabulate import tabulate

#pip install openpyxl 
# %%
#------Some settings for plotting and paths--------#
mpl.rcParams['figure.dpi'] = 600
base_dir = os.getcwd()
# print("Current working directory: ", base_dir)
# %matplotlib inline

#-------Import .MAT file into a dataframe--------#
# Load train and test data

train = pd.read_csv(os.path.join(base_dir, "train.csv"))
test = pd.read_csv(os.path.join(base_dir, "test.csv"))
'''
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
'''
original_test = test.copy()
# %%

#-----Trying some EDA to understand the data------#
train.head()
train.dtypes
train.columns
train.describe()

# %%
#-----Visualizing features for further processing---------#
features_to_inspect = ['Item_Weight', 'Item_Visibility', 'Item_MRP']

'''
for feature in features_to_inspect:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=data[feature])
    plt.title(f'Boxplot of {feature}')
    plt.subplot(1, 2, 2)
    sns.histplot(data[feature], bins=30, kde=True) #Kernel Density Estimate (KDE) curve shows the smoothed probability density of the data
    plt.title(f'Histogram of {feature}')
    plt.tight_layout()
    plt.show()
'''
# %%
#------Based on EDA and visual analysis, doing some data processing------#
train.isnull().sum()

for col in ['Outlet_Size']:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    test[col] = test[col].fillna(mode)
for col in ['Item_Weight']:
    mean = train[col].mean()
    train[col] = train[col].fillna(mean)
    test[col] = test[col].fillna(mean)

# Limiting outliers in train data for Item_visibility
lower = train['Item_Visibility'].quantile(0.01)
upper = train['Item_Visibility'].quantile(0.99)
train['Item_Visibility'] = train['Item_Visibility'].clip(lower, upper)

# Limiting outliers in test data for Item_visibility
lower = test['Item_Visibility'].quantile(0.01)
upper = test['Item_Visibility'].quantile(0.99)
test['Item_Visibility'] = test['Item_Visibility'].clip(lower, upper)

# Checking if any rows have null cells
test.isnull().sum()

#------Standardize Item_Fat_Content values before encoding------
fat_map = {
    'low fat': 'Low Fat', 
    'LF': 'Low Fat', 
    'Low Fat': 'Low Fat', 
    'Low fat': 'Low Fat',
    'reg': 'Regular',
    'Regular': 'Regular',
    'regular': 'Regular'
}

train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(fat_map)
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(fat_map)
# %%

#-----Label Encoding for high-cardinality columns-----#
le_item = LabelEncoder()
le_outlet = LabelEncoder()
train['Item_Identifier'] = le_item.fit_transform(train['Item_Identifier'])
test['Item_Identifier'] = le_item.transform(test['Item_Identifier'])
train['Outlet_Identifier'] = le_outlet.fit_transform(train['Outlet_Identifier'])
test['Outlet_Identifier'] = le_outlet.transform(test['Outlet_Identifier'])

train['Item_Outlet_Sales'] = pd.to_numeric(train['Item_Outlet_Sales'], errors='coerce')

#------One-hot encoding for low-cardinality columns------#
cat_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
full = pd.concat([train[cat_cols], test[cat_cols]], axis=0)
full_encoded = pd.get_dummies(full, columns=cat_cols, dtype=int)
train_encoded = full_encoded.iloc[:len(train), :].reset_index(drop=True)
test_encoded = full_encoded.iloc[len(train):, :].reset_index(drop=True)

#-----Creating interaction features-----#
train['MRP_Visibility_Interaction'] = train['Item_MRP'] * train['Item_Visibility']
test['MRP_Visibility_Interaction'] = test['Item_MRP'] * test['Item_Visibility']

# %%

# Merge encoded columns back
df_train = pd.concat([train.drop(cat_cols, axis=1).reset_index(drop=True), train_encoded], axis=1)
df_test = pd.concat([test.drop(cat_cols, axis=1).reset_index(drop=True), test_encoded], axis=1)

df_train_infer = df_train.drop(['Item_Outlet_Sales'], axis =1)
df_train_Target = df_train[['Item_Outlet_Sales']]

#-----Visualizing heatmap to inspect for any feature feature engineering-----#
sns.heatmap(df_train.corr(), cmap="BrBG", annot = False, annot_kws={"size":16}, square = True)
# Heatmap shows Item_type as less correlated with rest features, however sometimes they can add value to XG Boost with hidden or non-linearity fixing. So, I'm prcoeeding without any feature removal

# %%
from xgboost import XGBRegressor

XGB_model = XGBRegressor(booster = 'gbtree', eta = 0.01, max_depth = 5, 
                         objective = 'reg:squarederror', #[default=reg:squarederror]
                         max_delta_step =0, subsample =0.8, 
                         n_estimators=1000, learning_rate=0.01, 
                         colsample_bytree =1, alpha=0.6, gamma=0.4, 
                         reg_alpha=1, # L1 regularization
                         reg_lambda=1, # L2 regularization
                         tree_method = 'exact') #For large datasets, tree_method='hist' is faster.
#updater = 'prune', max_leaf_nodes =, scale_pos_weight =,  )

history_XGB = XGB_model.fit(df_train_infer, df_train_Target)

prediction_XGB = XGB_model.predict(df_test)
prediction_XGB = np.clip(prediction_XGB, 0, None)

# %%

#------Export the output as a CSV file------#
output = pd.DataFrame({
    'Item_Identifier': original_test['Item_Identifier'],
    'Outlet_Identifier': original_test['Outlet_Identifier'],
    'Item_Outlet_Sales': prediction_XGB
})

# output.to_csv("output.csv", index=False)
output.to_csv(os.path.join(base_dir, "output.csv"), index=False)

# %%

