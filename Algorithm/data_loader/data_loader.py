import h5py
import numpy as np


class DataLoader:
    def __init__(self, is_dev=False):

        self.is_dev=is_dev

        # testing data
        self.testing = self._open_h5_('/Users/chenjialu/Desktop/DL_Assignment1/Assignment-1-Dataset/test_128.h5', 'data')

        # training+validating data
        self.training = self._open_h5_('/Users/chenjialu/Desktop/DL_Assignment1/Assignment-1-Dataset/train_128.h5', 'data')
        self.label = self._open_h5_('/Users/chenjialu/Desktop/DL_Assignment1/Assignment-1-Dataset/train_label.h5', 'label')


    """
    function to do split the data
    """
    def split_train(self):
        # split the training and validation data
        n = np.random.np.random.randint(0, len(self.label), 60000)
        i = 50000
        # training data
        self.training_dev = self.training[n[:i], :]
        self.label_dev = self.label[n[:i]]
        # validating data
        self.training_val = self.training[n[i:], :]
        self.label_val = self.label[n[i:]]

    def load_data(self):
        return self

    @staticmethod
    def _open_h5_(filename, h_index):
        with h5py.File(filename, 'r') as H:
            data = np.copy(H[h_index])

            return data







