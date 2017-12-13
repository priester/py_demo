from sklearn.cross_validation import train_test_split
from sklearn.linear_model import  RidgeCV
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.12f' % x)

data=pd.read_csv("AmesHouse_FE_train.csv")

print(type(data))
print(data.head())
print(data.info())
print(data.shape) 
print (data.isnull().sum())
print(data.describe())




y = data['SalePrice'].values
X = data.drop('SalePrice', axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.1)



alphas = [0.01, 0.1, 1, 10,20, 40, 80,100]
reg = RidgeCV(alphas=alphas, store_cv_values=True)   
reg.fit(X_train, y_train)  

mse_mean = np.mean(reg.cv_values_, axis = 0)
plt.plot(np.log10(alphas), mse_mean.reshape(len(alphas),1)) 
plt.plot(np.log10(reg.alpha_)*np.ones(3), [0.28, 0.29, 0.30])
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print ('alpha is:', reg.alpha_)
reg.coef_

print 'The value of default measurement of RidgeRegression is', reg.score(X_test, y_test)