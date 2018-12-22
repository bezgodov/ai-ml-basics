import numpy as np

# function value
def linear_func(theta, x):
    return (x * theta).sum()

# 1-d np.array of function values of all rows of the matrix X
def linear_func_all(theta, X):
    return np.dot(X, theta)

# MSE value of current regression
def mean_squared_error(theta, X, y):
    i = 0
    score = 0
    lf = linear_func_all(theta, X)
    for _ in X:
        score += (y[i] - lf[i]) ** 2
        i += 1

    return score / i

# 1-d array of gradient by theta
def grad_mean_squared_error(theta, X, y):
    i = 0
    score = 0.0
    lf = linear_func_all(theta, X)
    for val in X:
        score += (lf[i] - y[i]) * val
        i += 1
    score *= 2 / i

    return score