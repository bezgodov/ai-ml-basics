import numpy as np

class GradientOptimizer:
	def __init__(self, oracle, x0):
		self.oracle = oracle
		self.x0 = x0

	def optimize(self, iterations, eps, alpha):
		x_pre = x_cur = self.x0
		for _ in range(iterations):
			x_cur = x_pre - alpha * self.oracle.get_grad(x_pre)

			if (abs((x_cur - x_pre).all()) <= eps):
				break
				
			x_pre = x_cur

		return x_cur