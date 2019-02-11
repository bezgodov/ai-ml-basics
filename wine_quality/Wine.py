import numpy as np
import pandas as pd
from datetime import datetime
from os import path

from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier

# def fullPath(file):
#     fld = path.dirname(path.abspath(__file__))
#     return path.join(fld, file)

class Wine:
    def __init__(self):
        self.seed = 13893
        np.random.seed(self.seed)

    def fit(self, X, y):
        clf1 = DecisionTreeClassifier(min_samples_split=2, random_state=self.seed)
        clf2 = QuadraticDiscriminantAnalysis(reg_param=0, tol=1.0e-8)
        clf3 = ExtraTreesClassifier(n_estimators=103, warm_start=True, random_state=self.seed)

        self.eclf = VotingClassifier(estimators=[
            ('dtc', clf1),
            ('qda', clf2),
            ('etc', clf3),
        ], voting='hard')

        self.eclf.fit(X, y)
    def predict(self, X):
        return self.eclf.predict(X)

# df = pd.read_csv(fullPath('wine_X_train.csv')).drop(columns=["Unnamed: 0"])
# X = x_df = df.drop(columns=["quality"])
# y = y_df = np.where(df['quality'] > 6, 1, 0)

# X_test = pd.read_csv(fullPath('wine_X_test.csv')).drop(columns=["Unnamed: 0"])

# wn = Wine()
# wn.fit(X, y)
# res = wn.predict(X=X_test)
# open(fullPath('results/' + str(datetime.now()) + '.txt'), 'w').write(' '.join(map(str, res)))
