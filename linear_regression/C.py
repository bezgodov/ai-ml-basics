import numpy as np

def fit_linear_regression(X, y):
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

# def fit_linear_regression(X, y):
#     alpha = 0.06

#     theta = np.ones_like(X[0,:])
#     grad = grad_mean_squared_error(theta, X, y)
#     error = mean_squared_error(theta, X, y)
#     while error > 0.0000000001:
#         theta -= grad * alpha
#         grad = grad_mean_squared_error(theta, X, y)
#         error = mean_squared_error(theta, X, y)

#     return theta

# X = np.array([
#     [1.0, 10.0, 12.0],
#     [1.0, 15.0, 10.0],
#     [1.0, 20.0, 9.0],
#     [1.0, 25.0, 9.0],
#     [1.0, 40.0, 8.0],
#     [1.0, 37.0, 8.0],
#     [1.0, 43.0, 6.0],
#     [1.0, 35.0, 4.0],
#     [1.0, 38.0, 4.0],
#     [1.0, 55.0, 5.0],
# ])
# y = np.array(
#     [
#         20.0,
#         35.0,
#         30.0,
#         45.0,
#         60.0,
#         69.0,
#         75.0,
#         90.0,
#         105.0,
#         110.0,
#     ]
# )
# print(fit_linear_regression(X, y))