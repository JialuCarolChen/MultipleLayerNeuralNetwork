import numpy as np

class Activation(object):

    def __init__(self, activation='tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_derivative
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'softmax':
            self.f = self.__softmax
            self.f_deriv = self.__softmax_derivative
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_derivative
        elif activation == None:
            self.f = None
            self.f_deriv = None

    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a ** 2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_derivative(self, a):
        # a = logistic(x)
        return a * (1 - a)

    def __relu(self, x):
        return np.maximum(0, x)

    def __relu_derivative(self, a):
        # a = relu(x)
        return np.array([[1 if y > 0 else 0 for y in x] for x in a])
        # return np.array([1 if y > 0 else 0 for y in a])

    def __softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / (np.sum(exps, axis=1, keepdims=True) + 1e-11) + 1e-11

    def __softmax_derivative(self, a):
        # a = softmax(x)
        return np.array([i - i**2 for i in a])
