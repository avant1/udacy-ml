import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(0)

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

input_size = len(X[0])
classifier = Sequential()

# classifier.add(BatchNormalization(input_dim=input_size, output_dim=input_size))

# two hidden layers
classifier.add(Dense(input_dim=input_size, output_dim=6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# output layer
# use output_dim=N_classes and 'softmax' activation for multi-class classification
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# use loss='categorical_crossentropy' for multi-class classification
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=5)

y_pred_probabilities = classifier.predict(X_test, verbose=1)

print(y_pred_probabilities)

threshold = 0.5
y_pred: np.ndarray = (y_pred_probabilities > threshold)
y_pred = y_pred.astype(int)

print(confusion_matrix(y_test, y_pred))

report = classification_report(y_test, y_pred)
print(report)
