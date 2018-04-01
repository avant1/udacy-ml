# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv').as_matrix()
X = dataset[:, [2, 3]]
y = dataset[:, 4].astype(int)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# applying k-fold cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_test, y_test, cv=10)

print(scores)
print('Average: {:.3f}'.format(scores.mean()))
print('Median: {:.3f}'.format(np.median(scores)))
print('Standart deviation: {:.3f}'.format(np.std(scores)))
