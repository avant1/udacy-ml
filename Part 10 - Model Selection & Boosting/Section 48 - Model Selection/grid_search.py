import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':

    # Importing the dataset
    dataset = pd.read_csv('Social_Network_Ads.csv').as_matrix()
    X = dataset[:, [2, 3]]
    y = dataset[:, 4].astype(int)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Kernel SVM to the Training set
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # applying k-fold cross validation
    scores = cross_val_score(classifier, X_test, y_test, cv=5)

    print('Linear SVM classifier results')
    print(scores)
    print('Average: {:.5f}'.format(scores.mean()))
    print('Median: {:.5f}'.format(np.median(scores)))
    print('Standart deviation: {:.5f}'.format(np.std(scores)))

    # grid search
    from sklearn.model_selection import GridSearchCV

    possible_parameters = {
        'C': [1, 2, 5],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4, 5],
        'gamma': [0.1, 0.3, 0.5, 0.7, 1, 'auto'],
        'coef0': [0.0, 0.1, 0.3],
    }

    classifier = GridSearchCV(classifier, possible_parameters, scoring='f1', cv=10, n_jobs=-1)
    classifier.fit(X_train, y_train)

    print('Optimal parameters:')
    print(classifier.best_params_)

    best_classifier = classifier.best_estimator_
    scores = cross_val_score(best_classifier, X_test, y_test, cv=5)

    print('Optimal SVM classifier results')
    print(scores)
    print('Average: {:.5f}'.format(scores.mean()))
    print('Median: {:.5f}'.format(np.median(scores)))
    print('Standart deviation: {:.5f}'.format(np.std(scores)))
