import numpy as np


class WeightInit:

    def __init__(self, method=None, n_in=None, n_out=None):
        """
        A class to perform weight initialization

        :type method: str
        :param method: the method to initialize the weights

        :type n_in: int
        :param n_in: the size of the row dimension of the weights

        :type n_out: int
        :param n_out: the size of the column dimension of the weights

        """

        self.W = None
        if method == None:
            setattr(self, "w", self.random_init(n_in, n_out))

        if method == "Kaiming":
            # the Kaiming method
            gain = 1
            std = gain * np.sqrt(1.0 / n_in)
            a = np.sqrt(3.0) * std
            setattr(self, "w", np.random.uniform(-a, a))

        elif method == "He":
            # the he method
            setattr(self, "w", np.random.randn(n_in, n_out) / np.sqrt(n_in / 2))

        elif method == "Xavier":
            # the Xavier method
            setattr(self, "w", np.random.normal(
                loc=0,
                scale=np.math.sqrt(2.0 / n_in),
                size=(n_in, n_out)))

        else:
            raise Exception('Method no valid!')


    def get_weights(self):
        """A function to return the weights"""
        return getattr(self, "w")

    def random_init(self, n_int, n_out):
        rn = np.random.RandomState(1234)
        w = rn.uniform(
            low=-np.sqrt(6. / (n_int + n_out)),
            high=np.sqrt(6. / (n_int + n_out)),
            size=(n_int, n_out)

        )
        return w
