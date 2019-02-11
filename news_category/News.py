import numpy as np
import pandas as pd
from datetime import datetime
from os import path

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# def fullPath(file):
#     fld = path.dirname(path.abspath(__file__))
#     return path.join(fld, file)

class News:
    def __init__(self):
        self.seed = 0
        np.random.seed(self.seed)
    def fit(self, X, y):
        self.vect = CountVectorizer(analyzer='word', stop_words=stop_words, ngram_range=(1, 2), lowercase=True, strip_accents=None, binary=True)
        _X = self.vect.fit_transform(X)
        self.clf = LinearSVC(C=0.2540, class_weight=None, dual=True, fit_intercept=True,
                            intercept_scaling=7, loss='hinge', max_iter=100,
                            multi_class='ovr', penalty='l2', random_state=self.seed, tol=1e-04, verbose=0)
        self.clf.fit(_X, y)

    def predict(self, X):
        return self.clf.predict(self.vect.transform(X))

# df = pd.read_csv(fullPath('news-train.csv'))
# test_df = pd.read_csv(fullPath('test-full.csv'))

# y = df['CAT'].values
# X = df['HEADER'].values
# X_test = test_df['HEADER'].values

# nw = News()
# nw.fit(X, y)
# res = nw.predict(X_test)
# open(fullPath('results/[LSVC]' + str(datetime.now()) + '.txt'), 'w').write('\n'.join(res))
