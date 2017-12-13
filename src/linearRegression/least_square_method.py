import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


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

# ss_X = StandardScaler()
# ss_y = StandardScaler()
# 
# 
# X_train = ss_X.fit_transform(X_train)
# X_test = ss_X.transform(X_test)



# y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
# y_test = ss_y.transform(y_test.reshape(-1, 1))
# 
# ss_X = StandardScaler()
# ss_y = StandardScaler()
# X_train = ss_X.fit_transform(X_train)
# X_test = ss_X.transform(X_test)
# 
# y_train = ss_y.fit_transform(y_train)
# y_test = ss_y.transform(y_test)
# y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
# y_test = ss_y.transform(y_test.reshape(-1, 1))


from sklearn.linear_model import LinearRegression


lr = LinearRegression()
lr.fit(X_train, y_train)


lr_y_predict = lr.predict(X_test)
lr_y_predict_train = lr.predict(X_train)


lr.coef_

print(lr.coef_)

print 'The value of default measurement of LinearRegression on test is', lr.score(X_test, y_test)
print 'The value of default measurement of LinearRegression on train is', lr.score(X_train, y_train)


f, ax = plt.subplots(figsize=(7, 5)) 
f.tight_layout() 
ax.hist(y_train - lr_y_predict_train,bins=400, label='Residuals Linear', color='b', alpha=.5); 
ax.set_title("Histogram of Residuals") 
ax.legend(loc='best');
plt.show()



print y_train
print lr_y_predict_train

plt.figure(figsize=(4, 4))
plt.scatter(y_train, lr_y_predict_train)
plt.plot([-3, 3], [-3, 3], '--k')   
plt.axis('tight')
plt.xlabel('True price')
plt.ylabel('Predicted price')
plt.tight_layout()
plt.show()