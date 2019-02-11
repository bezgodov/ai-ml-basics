import numpy as np

def sigmoid(t):
   return 1.0 / (1.0 + np.exp(-t))

def cross_entropy_loss(theta, X, y):
   return np.log(1.0 + np.exp(-y * (theta.dot(X.T)))).mean()

def fit_logistic_regression(X, y):
    weights = np.ones(X.shape[1])
    learning_rate = 1e15
    # output_error_signal = y - sigmoid(np.dot(X, weights))
    # for _ in range(200000):
    while 0.00001 < cross_entropy_loss(weights, X, y):
        # scores = np.dot(X, weights)
        # predictions = sigmoid(scores)

        # Update weights with gradient
        # output_error_signal = y - predictions
        gradient = grad_cross_entropy_loss(weights, X, y)
        weights -= learning_rate * gradient
        
    return weights

def grad_cross_entropy_loss(theta, X, y):
    return -np.mean((X.T * np.tile(y / ( 1.0 + np.exp(y * X.dot(theta))), (X.shape[1], 1))).T, axis=0)
# def fit_logistic_regression(X, y):
#     a = 17
#     theta = np.ones(X.shape[1])
#     error = 0.00001
#     while error < cross_entropy_loss(theta, X, y):
#         theta -= a * grad_cross_entropy_loss(theta, X, y)
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
# print(fit_logistic_regression(X, y))
