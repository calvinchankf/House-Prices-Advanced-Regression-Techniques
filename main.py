import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# TODO:
# feature encoding: WIP
# remove outliers: TBD

class HousePrices():

  def __init__(self):
    self.model = None
    self.X_test = None
    self.Y_test = None

  def trainModel(self):
    train = pd.read_csv('./data/train.csv')

    train = self.extractNumeric(train)

    Y = np.log(train.SalePrice)
    X = train.drop(['SalePrice', 'Id'], axis=1)

    X_train, self.X_test, Y_train, self.Y_test = train_test_split(X, Y, random_state=42, test_size=.33)
    lr = linear_model.LinearRegression()
    self.model = lr.fit(X_train, Y_train)
    print ("R^2 is: \n", self.model.score(self.X_test, self.Y_test))

    return

  # input:  training set
  # output: training set only with numeric values
  # pure function
  def extractNumeric(self, data):
    return data.select_dtypes(include=[np.number]).interpolate().dropna()

  # pure function
  # do it generically with pandas handy method
  # problem: inconsistent columns
  # input:  training set
  # output: feature encoded training set
  def genericOneHotEncode(self, data):
    categoricals = data.select_dtypes(exclude=[np.number])
    categorical_columns = categoricals.columns.tolist()
    result = pd.get_dummies(train, columns=categorical_columns)
    return result
    
  # input:  training set
  # output: feature encoded training set
  # manual
  def oneHotEncode(self, data):
    # do it generically with pandas handy method
    # problem: inconsistent columns
    categoricals = data.select_dtypes(exclude=[np.number])
    categorical_columns = categoricals.columns.tolist()
    result = pd.get_dummies(train, columns=categorical_columns)
    return result
  
  # lets see which one of either one-hot or feature-encoding can do a better job
  def featureEncode(self, data):
    return

  # predict the test data by train_test_split
  def crossValidate(self):
    if self.X_test is None or self.Y_test is None:
      return
    predictions = self.model.predict(self.X_test)
    print ('RMSE is: \n', mean_squared_error(self.Y_test, predictions))
    # plot it
    plt.scatter(predictions, self.Y_test, alpha=.75, color='b') #alpha helps to show overlapping data
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Linear Regression Model')
    plt.show()
    return
  
  # predict data for submission
  def predict(self):
    data = pd.read_csv('./data/test.csv')

    submission = pd.DataFrame()
    submission['Id'] = data.Id

    data = self.extractNumeric(data)
    data = data.drop(['Id'], axis=1)

    predictions = self.model.predict(data)
    final_predictions = np.exp(predictions)
    print ("Final predictions are: \n", final_predictions[:5])

    submission['SalePrice'] = final_predictions
    submission.to_csv('./submission2.csv', index=False)
    return

hp = HousePrices()
hp.trainModel()
hp.crossValidate()