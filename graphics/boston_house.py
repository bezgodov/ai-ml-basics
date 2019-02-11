import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plot
import seaborn as sns

from os import path

def fullPath(file):
    fld = path.dirname(path.abspath(__file__))
    return path.join(fld, file)

# dataset = load_boston()
# x_all, y_all = dataset.data[0], dataset.target

df = pd.read_csv(fullPath('boston_house_prices.csv'), skiprows=(0,0))#, 'LSTAT', 'DIS', 'INDUS', 'CRIM', 'MEDV', 'AGE'])
X = df.drop(columns=['MEDV'], axis=1)
y = df['MEDV'].values
# a = 3
# b = 1
# y.append([a + b * i for i in range(len(df.values))])
# y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

clf = LinearRegression(normalize=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(r2_score(y_test, y_pred))
print(clf.coef_)

plt.figure(figsize=(5, 5))

cmap = plt.cm.get_cmap("hsv", len(y) + 1)
# for i, val in enumerate(X_train):
# plt.scatter(clf.predict(X_train), y_train, color='red', label='scatter-{0:1d}'.format(0 + 1))
# plt.scatter(clf.predict(X_train), y_train, color='red', label='Train data')
# plt.scatter(clf.predict(X_test), y_test, color='blue', label='Test data')
# plt.scatter(clf.predict(X_test), y_pred, color='yellow', label='Predicted data')
plt.plot(y_test)
plt.plot(y_pred)
# plt.plot(clf.predict(X_test), y_pred, color='green', linewidth=3)
# plt.plot(range(len(y_pred)), y_pred, color='black', label='predicted')
plt.legend()
plt.show()
# corr = df.corr().abs()
# f, ax1 = plt.subplots(figsize=(10, 10))
# sns.heatmap(corr, ax=ax1, annot=True)
# plt.show()
