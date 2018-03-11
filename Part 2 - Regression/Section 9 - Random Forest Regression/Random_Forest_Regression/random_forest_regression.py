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
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


n_trees_to_try = [10, 100, 300, 3000, 10000]

plt.scatter(X, y, color='red', label='Input dataset')
plt.title('Truth or Bluff (Random forest regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')


# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor

for n_trees in n_trees_to_try:
    regressor = RandomForestRegressor(n_estimators=n_trees, random_state=0, n_jobs=-1)
    regressor.fit(X, y)

    # Predicting a new result
    new_employee_level = 6.5
    y_pred = regressor.predict(new_employee_level)[0]
    print('For {} trees predicted salary is: {:.2f}'.format(n_trees, y_pred))

    # Visualising the Regression results (for higher resolution and smoother curve)
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.plot(X_grid, regressor.predict(X_grid), label='Predicted values (n_trees={})'.format(n_trees))
    plt.scatter(new_employee_level, y_pred, marker='x', label='Test value (n_trees={})'.format(n_trees))

plt.legend()
plt.show()
