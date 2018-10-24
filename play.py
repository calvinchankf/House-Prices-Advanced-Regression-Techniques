import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# step 1: visualization
# plot any one of the params(x1) against the real data(y)
train = pd.read_csv('./data/train.csv')
x = train['LotArea']
y = train['SalePrice']

# describe
sp = train['SalePrice']
print(sp.describe())
print ("Skew is:", sp.skew())

# extract the numeric factors
numeric_features = train.select_dtypes(include=[np.number])
print(numeric_features.dtypes)

# see correlation
corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])

# discrete value
print("Unique OverallQual", train.OverallQual.unique())
print("Unique MiscFeature", train.MiscFeature.unique())

# take a look at the median of y for each OverallQual discrete value
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
print(quality_pivot)
# plot it
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
# plt.show()

# check correlation between GrLivArea and Sale Price
plt.scatter(x=train['GrLivArea'], y=np.log(sp))
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
# plt.show()

# check correlation between GarageArea and Sale Price
plt.scatter(x=train['GarageArea'], y=np.log(sp))
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
# plt.show()

# remove outliers in the stupidest way 
temp = train[train['GarageArea'] < 1200]
plt.scatter(x=temp['GarageArea'], y=np.log(temp.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
# plt.show()

# checkout null
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# checkout non-numeric data
categoricals = train.select_dtypes(exclude=[np.number])
print("🙄", categoricals.describe())
print("🤔", categoricals.columns.tolist())
categorical_columns = categoricals.columns.tolist()
# print("🏞", categoricals.iloc[:1,:].values.tolist())

# feature-encode the value of "Street"
# Pave => 1
# Grav => 0
# train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
# print (train.enc_street.value_counts())

# feature-encode the value of "MSZoning"
train = pd.get_dummies(train, columns=categorical_columns, drop_first=True)
print(train)
# print(train['enc_street'])