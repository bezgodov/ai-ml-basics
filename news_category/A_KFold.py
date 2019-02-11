import numpy as np
import pandas as pd
from datetime import datetime
# import nltk
# import nltk.tokenize

# from nltk.stem import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer
# from nltk.corpus import stopwords
# import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

from pathlib import Path

def file_location(file):
    script_location = Path(__file__).absolute().parent
    return script_location / file

df = pd.read_csv(file_location('news-train.csv'))
X = df['HEADER'].values

c = []
for x in X:
    try:
        x.encode('ascii')
    except UnicodeEncodeError as err:
        c.append(x)
print('WRONG:', len(c))

y = df['CAT'].values

vect = CountVectorizer(analyzer='word', stop_words='english')
X = vect.fit_transform(X)

kf = KFold(n_splits=2)
kf.get_n_splits(X)

KFold(n_splits=2, random_state=None, shuffle=False)

test_df = pd.read_csv(file_location('test-full.csv'))
x_test = vect.transform(test_df['HEADER'].values)


res = kf.split(x_test)
for train_index, test_index in res:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(y_test)

# open(file_location('results/[LSVC]' + str(datetime.now()) + '.txt'), 'w').write('\n'.join(y_pred))
