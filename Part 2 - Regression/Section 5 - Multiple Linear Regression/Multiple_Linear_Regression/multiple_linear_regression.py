# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# for some reason statsmodels does not work well when converting read data to nparray
# so using this kind of data load
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:, 3] = LabelEncoder().fit_transform(X[:, 3])
X = OneHotEncoder(categorical_features=[3]).fit_transform(X).toarray()

# removing one dummy variable
# cause it's presence does not bring any new information
# however LinearRegression will perform redundancy check automatically
# and remove one of those columns
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
score = regressor.score(X_test, y_test)
print('Score is: {:.3f}'.format(score))

predicted = regressor.predict(X_test)

diff = predicted - y_test
plt.hist(diff.reshape(-1, 1))
plt.title('Errors of predictions for test set')
plt.show()

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

# ols - ordered least squares
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_ols.summary())

X_opt = X[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_ols.summary())

X_opt = X[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_ols.summary())

X_opt = X[:, [0, 3]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_ols.summary())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
score_after_stepway_regression = regressor.score(X_test, y_test)
print('Score after applying stepway regression is: {:.3f}'.format(score_after_stepway_regression))

print('Wow, we\'ve just got {:.5f} score improvement!!1'.format(score_after_stepway_regression - score))
print('Stepway regression is considered harmful')
print('When the number of features is a problem - use PCA')
