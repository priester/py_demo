from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoCV

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


pd.set_option('display.float_format', lambda x: '%.12f' % x)

data = pd.read_csv("AmesHouse_FE_train.csv")

print(type(data))
print(data.head())
print(data.info())
print(data.shape) 
print (data.isnull().sum())
print(data.describe())

y = data['SalePrice'].values
X = data.drop('SalePrice', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.1)

alphas = [0.01, 0.1, 1, 10, 100]

lasso = LassoCV(alphas=alphas)   
lasso.fit(X_train, y_train)    

mses = np.mean(lasso.mse_path_, axis=1)
plt.plot(np.log10(lasso.alphas_), mses) 
# plt.plot(np.log10(lasso.alphas_)*np.ones(3), [0.3, 0.4, 1.0])
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()    
            
print ('alpha is:', lasso.alpha_)
lasso.coef_  


print 'The value of default measurement of Lasso Regression on test is', lasso.score(X_test, y_test)
print 'The value of default measurement of Lasso Regression on train is', lasso.score(X_train, y_train)