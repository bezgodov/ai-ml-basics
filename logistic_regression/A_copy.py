import numpy as np

def logistic_func(theta, x):
   # return 1.0/(1.0+np.exp(-x.dot(theta)))
   return 1. / (1. + np.exp(-x.dot(theta)))
 
def logistic_func_all(theta, X):
   return np.fromiter((logistic_func(theta, x) for x in X), float)
 
def cross_entropy_loss(theta, X, y):
   return np.log(1. + np.exp(-y * (theta.dot(X.T)))).mean()
 
def grad_cross_entropy_loss(theta, X, y):
   return np.mean(np.tile(-y / (1. + np.exp(y * (theta.dot(X.T)))), (X.shape[1], 1)).T * X, axis=0)

# X = np.array([[1,2],[3,4],[4,5]])

# theta = np.array([5.0, 6.0])

# y = np.array([1, 2, 1])

# print(logistic_func_all(theta, X))

# print(cross_entropy_loss(theta, X, y))

# print(grad_cross_entropy_loss(theta, X, y))
