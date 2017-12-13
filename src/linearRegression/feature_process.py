# from IPython.display import display
# from scipy.stats import skew

from linearRegression import process_value
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
# import seaborn as sns


# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train = pd.read_csv("Ames_House_train.csv")
# train.info()


# print(train.head())
# print(train.describe())

test = pd.read_csv("Ames_House_test.csv")
# test.info()

categorical_features = train.select_dtypes(include = ["object"]).columns
for col in categorical_features:
    print '\n%s count'%col
    print train[col].value_counts()
    
train.drop(['Id'], inplace = True, axis = 1)
test_id = test['Id']
test.drop(['Id'], inplace = True, axis = 1)    

# plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")
# plt.title("Looking for outliers")
# plt.xlabel("GrLivArea")
# plt.ylabel("SalePrice")
# plt.show()

train = train[train.GrLivArea < 4000]
temp = train.reindex()
   
train = process_value.process_missvalue_by_meaning(train)
train = process_value.numberical2cat(train)
train = process_value.cat2numberical(train)
train = process_value.simplify(train)
train = process_value.Combine(train)

test = process_value.process_missvalue_by_meaning(test)
test = process_value.numberical2cat(test)
test = process_value.cat2numberical(test)
test = process_value.simplify(test)
test = process_value.Combine(test)

print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
#print(corr.SalePrice)

threshold = corr.SalePrice.iloc[11]  #the first one is SalePrice itself,from 1-11
print threshold
top10_cols = (corr.SalePrice[corr['SalePrice']>threshold]).axes

train = process_value.Polynomials_top10(train, top10_cols)
test = process_value.Polynomials_top10(test,top10_cols)

train_num, medians, ss_X = process_value.fillna_numerical_train(train)
# train_num.info()
test_num = process_value.fillna_numerical_test(test, medians, ss_X)

n_train_samples = train.shape[0]  
train_test = pd.concat((train, test), axis=0)
train_test_cat = process_value.get_dummies_cat(train_test)
   
train_cat = train_test_cat.iloc[:n_train_samples, :]
test_cat = train_test_cat.iloc[n_train_samples:, :]

# train_cat.info()

FE_train = process_value.joint_num_cat(train_num, train_cat)
FE_test = process_value.joint_num_cat(test_num, test_cat)

FE_train.info()

FE_train = pd.concat([FE_train, train['SalePrice']], axis = 1)
FE_test = pd.concat([test_id,FE_test], axis = 1)

FE_train.to_csv('AmesHouse_FE_train.csv', index=False)
FE_test.to_csv('AmesHouse_FE_test.csv', index=False)

# FE_train.info()

# from sklearn.preprocessing import StandardScaler

# train.info()


# 
y = train['SalePrice'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(FE_train, y, random_state=33, test_size=0.2)

X_train.info()
X_test.info()


# 
# ss_X = StandardScaler()
# ss_y = StandardScaler()
# 
# 
# train = ss_X.fit_transform(train)
# test = ss_X.transform(test)
# 
# #y_train = ss_y.fit_transform(y_train)
# #y_test = ss_y.transform(y_test)
# 
# y_train = ss_y.fit_transform(train_y.reshape(-1, 1))
# y_test = ss_y.transform(test_y.reshape(-1, 1))






