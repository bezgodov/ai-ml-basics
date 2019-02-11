import numpy as np
import pandas as pd
import random
from datetime import datetime
# import nltk
# import nltk.tokenize

# from nltk.stem import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
# import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
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


y = df['CAT'].values
X = df['HEADER'].values


# foo = df2['HEADER'].values
# i_rem = []
# for i, x in enumerate(foo):
#     try:
#         x.encode('ascii')
#     except UnicodeEncodeError as err:
#         i_rem.append(x)

# y = np.delete(y, i_rem)
# X = np.delete(X, i_rem)

vect = CountVectorizer(analyzer='word', stop_words=stop_words, ngram_range=(1, 2), lowercase=True, strip_accents=None, binary=True)

# vect = TfidfVectorizer(input='content', encoding='utf-8',
#                  decode_error='strict', strip_accents=None, lowercase=True,
#                  preprocessor=None, tokenizer=None, analyzer='word',
#                  stop_words=stop_words, token_pattern=r"(?u)\b\w\w+\b",
#                  ngram_range=(1, 2), max_df=1.0, min_df=0.000001,
#                  max_features=None, vocabulary=None, binary=False,
#                  dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)


X = vect.fit_transform(X)

clf = LinearSVC(C=0.2540, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=7, loss='hinge', max_iter=100,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-04, verbose=0)

# BEST SCORE
# clf = LinearSVC(C=0.26, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=100,
#     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)

scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)

# grid = {'C':np.logspace(-5, 5, 11)}
# cv = KFold(n_splits=5, shuffle=True, random_state=742)

# clf = SVC(kernel='linear', random_state=742)
# gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)


print(scores.mean() * 10e4)
print(scores * 10e4)

clf.fit(X, y)

test_df = pd.read_csv(file_location('test-full.csv'))
x_test = vect.transform(test_df['HEADER'].values)
y_pred = clf.predict(x_test)

open(file_location('results/[LSVC]' + str(datetime.now()) + '.txt'), 'w').write('\n'.join(y_pred))
