# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv').as_matrix()
X = dataset[:, :-1]
y = dataset[:, -1]

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
