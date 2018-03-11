# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv').as_matrix()
X = dataset[:, 1:-1]
y = dataset[:, -1]

# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X.astype(float))
sc_y = StandardScaler()
y_scaled = sc_y.fit_transform(y.astype(float).reshape(-1, 1))

# Fitting the Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf', C=3)
regressor.fit(X_scaled, y_scaled)

y_pred_scaled = regressor.predict(sc_X.transform(6.5))
y_pred = sc_y.inverse_transform(y_pred_scaled)
print(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color='red')
predicted_y = sc_y.inverse_transform(regressor.predict(X_scaled))
plt.plot(X, predicted_y, color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
