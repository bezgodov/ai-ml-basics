import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LinearRegression, LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt
# %matplotlib inline
import os
def fullPath(file):
    fld = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(fld, file)

seed = 13893
np.random.seed(seed)

df = pd.read_csv(fullPath('wine_X_train.csv')).drop(columns=["Unnamed: 0"])
x_df = df.drop(columns=["quality"])
y_df = np.where(df['quality'] > 6, 1, 0)

def heatMap():
    corr = x_df.corr().abs()
    f, ax1 = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr, ax=ax1, annot=True)
    plt.show()
# heatMap()

def barPlot():
    kbest = SelectKBest(chi2, k=5)
    fit = kbest.fit(x_df, y_df)
    scores = kbest.scores_
    p_sort = np.argsort(scores)[::-1]
    f, ax1 = plt.subplots(figsize=(20, 5))
    sns.barplot(x=x_df.columns[p_sort], y=scores[p_sort], ax=ax1)
    plt.show()
# barPlot()

def LR():
    clf = LogisticRegression(solver='lbfgs')
    rfe = RFE(clf, 3)
    rfe.fit(x_df, y_df)
    print(rfe.n_features_)
    print(rfe.support_)
    print(rfe.ranking_)
    print(x_df.columns[rfe.support_])
# LR()

# print('-----RFC----')

def RFC():
    clf = RandomForestClassifier(n_estimators=100)
    rfe = RFE(clf, 3)
    rfe.fit(x_df, y_df)
    print(rfe.n_features_)
    print(rfe.support_)
    print(rfe.ranking_)
    print(x_df.columns[rfe.support_])

    clf.fit(x_df, y_df)
    p_sort = np.argsort(clf.feature_importances_)[::-1]
    f, ax1 = plt.subplots(figsize=(20, 10))
    sns.barplot(x=x_df.columns[p_sort], y=clf.feature_importances_[p_sort], ax=ax1)
    plt.show()
# RFC()

test_df = pd.read_csv(fullPath('wine_X_test.csv')).drop(columns=["Unnamed: 0"])

# scaler = StandardScaler()
# pca = PCA()
# scores = pca.fit(scaler.fit_transform(x_df))

X = x_df
y = y_df

clf1 = RandomForestClassifier(n_estimators=100, random_state=seed)
clf2 = LogisticRegression(solver='lbfgs', random_state=seed)
clf3 = GaussianNB(priors=None)
clf4 = SVC(C=1.0, kernel='linear', random_state=seed)
clf5 = DecisionTreeClassifier(min_samples_split=2, random_state=seed)
clf6 = KNeighborsClassifier()
clf7 = GaussianProcessClassifier(random_state=seed)
clf8 = MLPClassifier(random_state=seed)
clf9 = AdaBoostClassifier(base_estimator=GradientBoostingClassifier(), random_state=seed)
clf10 = LinearDiscriminantAnalysis()
clf11 = QuadraticDiscriminantAnalysis(reg_param=0, tol=1.0e-8)
clf12 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=seed)
clf13 = ExtraTreesClassifier(n_estimators=103, warm_start=True, random_state=seed)

eclf1 = VotingClassifier(estimators=[
    # ('rfc', clf1),
    # ('lg', clf2),
    # ('gnb', clf3),
    # ('svc', clf4)
    ('dtc', clf5),
    # ('knc', clf6),
    # ('gpc', clf7),
    # ('mlpc', clf8),
    # ('abc', clf9),
    # ('lda', clf10),
    ('qda', clf11),
    # ('gbc', clf12),
    ('etc', clf13),

], voting='hard')

# eclf1.estimators_

scores = cross_val_score(eclf1, X, y, scoring='accuracy', cv=3)

print(scores)

eclf1.fit(X, y)

print(eclf1.score(X, y))

y_pred = eclf1.predict(test_df)

    # plt.semilogy(scores.explained_variance_, '--o')
    # plt.semilogy(scores.explained_variance_ratio_, '--o')
    # plt.show()
# SS()

open(fullPath('results/' + str(datetime.now()) + '.txt'), 'w').write(' '.join(map(str, y_pred)))
