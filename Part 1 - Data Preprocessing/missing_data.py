# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv').as_matrix()
X = dataset[:, :-1]
y = dataset[:, -1]


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(X[:,1:])
X[:, 1:] = imputer.transform(X[:, 1:])
