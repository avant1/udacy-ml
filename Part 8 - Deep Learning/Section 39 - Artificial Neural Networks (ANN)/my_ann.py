import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

np.random.seed(0)

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
categorical_columns = [1, 2]
for column in categorical_columns:
    labelencoder = LabelEncoder()
    X[:, column] = labelencoder.fit_transform(X[:, column])

onehotencoder = OneHotEncoder(categorical_features=[1], sparse=False)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

input_size = len(X[0])
classifier = Sequential()

# two hidden layers
classifier.add(Dense(input_dim=input_size, units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# output layer
# use units=N_classes and 'softmax' activation for multi-class classification
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# use loss='categorical_crossentropy' for multi-class classification
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=5)

y_pred_probabilities = classifier.predict(X_test, verbose=1)

threshold = 0.5
y_pred: np.ndarray = (y_pred_probabilities > threshold)
y_pred = y_pred.astype(int)

print(confusion_matrix(y_test, y_pred))

report = classification_report(y_test, y_pred)
print(report)
