import pandas as pd
import numpy as np


from scipy.stats import skew


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

import pandas as pd
import numpy as np

from scipy.stats import skew

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
# Definitions
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
# %matplotlib inline
# Definitions
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
# %matplotlib inline

# s = pd.Series([1,3,5,np.nan,6,8])
# print(s)

help(pd.set_option('display.float_format', lambda x: '%.3f' % x))

# train = pd.read_csv("Ames_House_train.csv")
# print(train.head())
# 
# print(train.describe())
# 
# categorical_features = train.select_dtypes(include = ["object"]).columns
# for col in categorical_features:
#     print '\n%scount'%col
#     print train[col].value_counts()
    
# plt.scatter(train.GrLivArea, train., c = "blue", marker = "s")
# plt.title("Looking for outliers")
# plt.xlabel("PoolQC")
# plt.ylabel("SalePrice")
# plt.show()    
# 
# train.info()  
# 
# 
# def process_missvalue_by_meaning (df):
#     # Alley : data description says NA means "no alley access"
#     df.loc[:, "Alley"] = df.loc[:, "Alley"].fillna("None")
# 
#     # BedroomAbvGr : NA most likely means 0
#     df.loc[:, "BedroomAbvGr"] = df.loc[:, "BedroomAbvGr"].fillna(0)
# 
#     # BsmtQual etc : data description says NA for basement features is "no basement"
#     df.loc[:, "BsmtQual"] = df.loc[:, "BsmtQual"].fillna("No")
#     df.loc[:, "BsmtCond"] = df.loc[:, "BsmtCond"].fillna("No")
#     df.loc[:, "BsmtExposure"] = df.loc[:, "BsmtExposure"].fillna("No")
#     df.loc[:, "BsmtFinType1"] = df.loc[:, "BsmtFinType1"].fillna("No")
#     df.loc[:, "BsmtFinType2"] = df.loc[:, "BsmtFinType2"].fillna("No")
#     df.loc[:, "BsmtFullBath"] = df.loc[:, "BsmtFullBath"].fillna(0)
#     df.loc[:, "BsmtHalfBath"] = df.loc[:, "BsmtHalfBath"].fillna(0)
#     df.loc[:, "BsmtUnfSF"] = df.loc[:, "BsmtUnfSF"].fillna(0)
# 
#     # CentralAir : NA most likely means No
#     df.loc[:, "CentralAir"] = df.loc[:, "CentralAir"].fillna("N")
# 
#     # Condition : NA most likely means Normal
#     df.loc[:, "Condition1"] = df.loc[:, "Condition1"].fillna("Norm")
#     df.loc[:, "Condition2"] = df.loc[:, "Condition2"].fillna("Norm")
# 
#     # EnclosedPorch : NA most likely means no enclosed porch
#     df.loc[:, "EnclosedPorch"] = df.loc[:, "EnclosedPorch"].fillna(0)
# 
#     # External stuff : NA most likely means average
#     df.loc[:, "ExterCond"] = df.loc[:, "ExterCond"].fillna("TA")
#     df.loc[:, "ExterQual"] = df.loc[:, "ExterQual"].fillna("TA")
# 
#     # Fence : data description says NA means "no fence"
#     df.loc[:, "Fence"] = df.loc[:, "Fence"].fillna("No")
# 
#     # FireplaceQu : data description says NA means "no fireplace"
#     df.loc[:, "FireplaceQu"] = df.loc[:, "FireplaceQu"].fillna("No")
#     df.loc[:, "Fireplaces"] = df.loc[:, "Fireplaces"].fillna(0)
# 
#     # Functional : data description says NA means typical
#     df.loc[:, "Functional"] = df.loc[:, "Functional"].fillna("Typ")
# 
#     # GarageType etc : data description says NA for garage features is "no garage"
#     df.loc[:, "GarageType"] = df.loc[:, "GarageType"].fillna("No")
#     df.loc[:, "GarageFinish"] = df.loc[:, "GarageFinish"].fillna("No")
#     df.loc[:, "GarageQual"] = df.loc[:, "GarageQual"].fillna("No")
#     df.loc[:, "GarageCond"] = df.loc[:, "GarageCond"].fillna("No")
#     df.loc[:, "GarageArea"] = df.loc[:, "GarageArea"].fillna(0)
#     df.loc[:, "GarageCars"] = df.loc[:, "GarageCars"].fillna(0)
# 
#     # HalfBath : NA most likely means no half baths above grade
#     df.loc[:, "HalfBath"] = df.loc[:, "HalfBath"].fillna(0)
# 
#     # HeatingQC : NA most likely means typical
#     df.loc[:, "HeatingQC"] = df.loc[:, "HeatingQC"].fillna("TA")
# 
#     # KitchenAbvGr : NA most likely means 0
#     df.loc[:, "KitchenAbvGr"] = df.loc[:, "KitchenAbvGr"].fillna(0)
# 
#     # KitchenQual : NA most likely means typical
#     df.loc[:, "KitchenQual"] = df.loc[:, "KitchenQual"].fillna("TA")
# 
#     # LotFrontage : NA most likely means no lot frontage
#     df.loc[:, "LotFrontage"] = df.loc[:, "LotFrontage"].fillna(0)
# 
#     # LotShape : NA most likely means regular
#     df.loc[:, "LotShape"] = df.loc[:, "LotShape"].fillna("Reg")
# 
#     # MasVnrType : NA most likely means no veneer
#     df.loc[:, "MasVnrType"] = df.loc[:, "MasVnrType"].fillna("None")
#     df.loc[:, "MasVnrArea"] = df.loc[:, "MasVnrArea"].fillna(0)
# 
#     # MiscFeature : data description says NA means "no misc feature"
#     df.loc[:, "MiscFeature"] = df.loc[:, "MiscFeature"].fillna("No")
#     df.loc[:, "MiscVal"] = df.loc[:, "MiscVal"].fillna(0)
# 
#     # OpenPorchSF : NA most likely means no open porch
#     df.loc[:, "OpenPorchSF"] = df.loc[:, "OpenPorchSF"].fillna(0)
# 
#     # PavedDrive : NA most likely means not paved
#     df.loc[:, "PavedDrive"] = df.loc[:, "PavedDrive"].fillna("N")
# 
#     # PoolQC : data description says NA means "no pool"
#     df.loc[:, "PoolQC"] = df.loc[:, "PoolQC"].fillna("No")
#     df.loc[:, "PoolArea"] = df.loc[:, "PoolArea"].fillna(0)
# 
#     # SaleCondition : NA most likely means normal sale
#     df.loc[:, "SaleCondition"] = df.loc[:, "SaleCondition"].fillna("Normal")
# 
#     # ScreenPorch : NA most likely means no screen porch
#     df.loc[:, "ScreenPorch"] = df.loc[:, "ScreenPorch"].fillna(0)
# 
#     # TotRmsAbvGrd : NA most likely means 0
#     df.loc[:, "TotRmsAbvGrd"] = df.loc[:, "TotRmsAbvGrd"].fillna(0)
# 
#     # Utilities : NA most likely means all public utilities
#     df.loc[:, "Utilities"] = df.loc[:, "Utilities"].fillna("AllPub")
# 
#     # WoodDeckSF : NA most likely means no wood deck
#     df.loc[:, "WoodDeckSF"] = df.loc[:, "WoodDeckSF"].fillna(0)
#     
#     return df
#     
# train = process_missvalue_by_meaning(train)  