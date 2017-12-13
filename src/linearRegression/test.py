import pandas as pd

# train = pd.read_csv("Ames_House_train.csv")
# train.info()

# pd.set_option('display.float_format', lambda x: '%.3f' % x)

train = pd.read_csv("Ames_House_train.csv")
train.info()
y_train = train['SalePrice'].values

print(y_train)


# print(train.head())
# print(train.describe())

# test = pd.read_csv("Ames_House_test.csv")
# test.info()

# y_test = test['SalePrice'].values

# data=pd.read_csv("boston_housing.csv")
# data.info()
# 
# y = data['MEDV'].values
# X = data.drop('MEDV', axis = 1)
# 
# print(y);

