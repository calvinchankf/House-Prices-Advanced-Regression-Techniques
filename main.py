import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

train = pd.read_csv('./data/train.csv')

# TODO:
# feature encoding: WIP
# remove outliers: TBD

# transform categorical data to numeric data, i.e. one-hot/feature-encoding
categoricals = train.select_dtypes(exclude=[np.number])
categorical_columns = categoricals.columns.tolist()
abc = pd.get_dummies(train, columns=categorical_columns)
print("🤡", abc.columns.tolist())
# only select the numeric type features (remove the non-numeric)
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

# split X and Y
y = np.log(data.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1) # remove SalePrice and Id from column(axis=1)
print("😅", len(X.columns.tolist()))

# split train and test
# random_state: guarantee data are always splited into the same train & test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))

# cross validation: use the model to predict the result of X_test
predictions = model.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75, color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
# plt.show()

test = pd.read_csv('./data/test.csv')

submission = pd.DataFrame()
submission['Id'] = test.Id

categoricals = test.select_dtypes(exclude=[np.number])
categorical_columns = categoricals.columns.tolist()
abc = pd.get_dummies(test, columns=categorical_columns)
print("🤡", abc.columns.tolist())
feats = test.select_dtypes(include=[np.number]).interpolate().dropna()
feats = feats.drop(['Id'], axis=1)
print("😅", len(feats.columns.tolist()))

predictions = model.predict(feats)

final_predictions = np.exp(predictions)

print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])

submission['SalePrice'] = final_predictions
submission.head()

submission.to_csv('./submission2.csv', index=False)