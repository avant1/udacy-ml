# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk

np.random.seed(0)
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

# stopwords_set = set(stopwords.words('english'))
# todo: for this dataset using empty stopwords list shows highest CV score
# stopwords_set = set()

# todo: but this one, without negative word-parts shows higher score than nltk 'english' list
# todo: but the dataset is too small, and empty stopwords set is correct for 1 or 2 reviews more
# todo: so probably it is better to use general approach with common stopwords
# todo: but negative word parts should be kept as ngrams of 2 words are used
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stopwords_set = ENGLISH_STOP_WORDS.difference(['not', 'no', 'nor', 'none', 'never', 'nothing', 'very'])

stemmer = PorterStemmer()
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if word not in stopwords_set]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500, ngram_range=(1, 2))
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
# classifier = RandomForestClassifier(n_estimators=30)
classifier = LinearSVC(C=0.2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

scores = cross_val_score(classifier, X, y)
print(cm)
print(scores.mean())
print(scores.std())
print(metrics.classification_report(y_test, y_pred, target_names=['0', '1']))
