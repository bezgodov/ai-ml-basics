import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from pathlib import Path

def file_location(file):
    script_location = Path(__file__).absolute().parent
    return script_location / file

# dataset = load_boston()
# x_all, y_all = dataset.data[0], dataset.target

fl_train = file_location('train.csv')
fl_test = file_location('test.csv')
df_train = pd.read_csv(fl_train)
df_test = pd.read_csv(fl_test)

df_train['x'] = df_train['x'].fillna(0.0)
df_train['y'] = df_train['y'].fillna(0.0)

x_train, y_train = df_train['x'].values, df_train['y'].values
x_test, y_test = df_test['x'].values, df_test['y'].values

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

clf = LinearRegression(normalize=False)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(r2_score(y_test, y_pred))
print(clf.coef_)

plt.figure(figsize=(5,5))
plt.scatter(x_train, y_train, color='red', label='data')
plt.scatter(x_test, y_pred, color='blue', label='predicted')
plt.legend()
plt.show()
