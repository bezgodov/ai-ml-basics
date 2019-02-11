import numpy as np
import pandas as pd
from datetime import datetime

import nltk
import nltk.tokenize

# from nltk.stem import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
# import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from pathlib import Path

def file_location(file):
    script_location = Path(__file__).absolute().parent
    return script_location / file

df = pd.read_csv(file_location('news-train.csv'))


X = df['HEADER'].values
y = df['CAT'].values

vect = CountVectorizer(analyzer='word', stop_words=stop_words)
X = vect.fit_transform(X)

clf = LogisticRegression(
                    C=1.18255, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                    penalty='l2', random_state=None, solver='liblinear', tol=0.00001,
                    verbose=0, warm_start=False)

scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
print(scores)

clf.fit(X, y)

# LogisticRegression(
#                     C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                     intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
#                     penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#                     verbose=0, warm_start=False)


test_df = pd.read_csv(file_location('test-full.csv'))
x_test = vect.transform(test_df['HEADER'].values)
y_pred = clf.predict(x_test)

open(file_location('results/[LR]' + str(datetime.now()) + '.txt'), 'w').write('\n'.join(y_pred))
