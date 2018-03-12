# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv').as_matrix()
X = dataset[:, 1:-1]
y = dataset[:, -1]


from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

plt.scatter(X, y, color='red', marker='x', label='Train data')
plt.plot(X, linear_regressor.predict(X), label='Predicted data (linear model)')
plt.xlabel('Position level')
plt.ylabel('Salary, $')

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))


for degree in range(2, 5):
    from sklearn.preprocessing import PolynomialFeatures
    polynomial_regressor = PolynomialFeatures(degree=degree, include_bias=True)
    X_with_polynoms = polynomial_regressor.fit_transform(X)

    # basically we can use old linear_regression
    # but let's not mix up models
    # and not use variables defined much earlier
    linear_regressor_for_polynomial_features = LinearRegression()
    linear_regressor_for_polynomial_features.fit(X_with_polynoms, y)

    X_grid_with_polynoms = polynomial_regressor.fit_transform(X_grid)
    plt.plot(
        X_grid,
        linear_regressor_for_polynomial_features.predict(X_grid_with_polynoms),
        # color='black',
        label='Predicted data (polynomial model, degree {})'.format(degree)
    )

plt.legend(loc='upper left')
plt.show()
