import numpy as np

# function value
def logistic_func(theta, x):
   #  return (x * theta).sum()
   return 1./(np.exp(-x.dot(theta)) + 1.)
    # return (x * theta).sum()

# 1-d np.array of function values of all rows of the matrix X
def logistic_func_all(theta, X):
   #  return np.dot(X, theta)
   return np.dot(X, theta)

# MSE value of current regression
def cross_entropy_loss(theta, X, y):
    # i = 0
    # score = 0
    # lf = logistic_func_all(theta, X)
    # for _ in X:
    #     score += (y[i] - lf[i]) ** 2
    #     i += 1
    # z = logistic_func_all(theta, X)

    # np.mean(np.log(1 + np.exp(-y * w.dot(z.T))))

   # return np.mean(np.log(1 + np.exp(-y * w.dot(z.T))))


   # epsilon = 0#1e-5
   # m = X.shape[0]
   # yp = expit(X @ theta)
   # cost = -1 * np.average(y * np.log(yp + epsilon) + (1 - y) * np.log(1 - yp + epsilon))
   # return cost

   # m = X.shape[0]
   # total_cost = -(1 / m) * np.sum(
   #    y * np.log(probability(theta, X)) + (1 - y) * np.log(
   #       1 - probability(theta, X)))
   # return total_cost
   z = np.dot(X, theta)
   h = sigmoid(z)
   return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
   # return np.sum(y*np.log(logistic_func(theta, X)) + (1 - y)*np.log(1. - logistic_func(theta, X)))
   # z = np.dot(X, theta)
   # h = sigmoid(z)
   # return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

   # if y == 1:
   #    return -log(yHat)
   #  else:
   #    return -log(1 - yHat)

   #  lam = 2. * len(X)
   #  Z = np.dot(X,theta)
   #  #calculate cost without regularization
   #  #shape of X is (m,n), shape of w is (n,1)
   #  J = (1./len(X)) * (-np.dot(y.T, np.log(phi(Z))) * np.dot((1-y.T),np.log(1 - phi(Z))))
   #  #add regularization
   #  #temporary weight vector
   #  w1 = theta #import copy to create a true copy
   #  w1[0] = 0
   #  J += (lam/(2.*len(X))) * np.dot(w1.T, theta)
   #  return J

    #np.mean(np.log(1 + np.exp(-y * w.dot(z.T))))
# 1-d array of gradient by theta
# def probability(theta, x):
#     # Returns the probability after passing through sigmoid
#     return sigmoid(logistic_func_all(theta, x))
def sigmoid(t):
   return 1.0 / (1.0 + np.exp(-t))

def grad_cross_entropy_loss(theta, X, y):
   # grad = 0
   # alpha = 1e-5
   # print(theta)
   # for x in X:
   #    m = x.shape[0]
   #    h = sigmoid(np.matmul(x, theta))
   #    grad += np.matmul(X.T, (h - y)) / m
   #    theta = theta - alpha * grad
   # print(theta)
   # return grad

   # lf = logistic_func(theta, X)
   # return -X.dot(y - logistic_func_all(theta, X))
   z = np.dot(X, theta)
   h = sigmoid(z)
   gradient = np.dot(X.T, (h - y)) / y.shape[0]
   return gradient
   # x = X
   # m = x.shape[0]
   # return (1 / m) * np.dot(X.T, sigmoid(logistic_func_all(theta, X)) - y)
   # lr = 0.0001

   # for _ in ((X)):
   #    z = np.dot(X, theta)
   #    h = sigmoid(z)
   #    gradient = np.dot(X.T, (h - y)) / y.size
   #    theta -= lr * gradient
   # return gradient
      # if(self.verbose == True and i % 10000 == 0):
      #    z = np.dot(X, self.theta)
      #    h = self.__sigmoid(z)
      #    print(f'loss: {self.__loss(h, y)} \t')

# X = np.array([[1,2],[3,4],[4,5]])

# theta = np.array([5.0, 6.0])

# y = np.array([1, 2, 1])

# print(logistic_func_all(theta, X))

# print(cross_entropy_loss(theta, X, y))

# print(grad_cross_entropy_loss(theta, X, y))
