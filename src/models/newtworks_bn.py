from src.models.weight_init import WeightInit
import matplotlib as plt
import numpy as np
import time
import math


class HiddenLayer(object):
    def __init__(self, n_in, n_out, W=None, b=None, is_last=False, dropout=0.5):

        # parameters
        self.n_out = n_out
        self.epsl = 1e-5
        self.input = None
        self.output = None
        self.dropout_p = dropout

        # parameters to learn
        self.W = WeightInit(method=W, n_in=n_in, n_out=n_out).get_weights()
        self.beta = None
        self.gamma = None

        # intermediate values
        self.lin_output = None
        self.mean = None
        self.x_norm = None
        self.var = None
        self.x = None

        # analytical gradient
        self.grad_W = np.zeros(self.W.shape)
        self.grad_beta = None
        self.grad_gamma = None

        # momentum
        self.v = np.zeros((n_in, n_out))

        # layer control
        self.is_last = is_last
        self.layer_mask = None

        # Running mean for validating/testing
        self.running_mean = None
        self.running_variance = None

    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''

        #save input
        self.input = input
        # initialize gamma and beta as the initial mean and std of WX
        if self.beta is None:
            #self.beta = np.mean(np.dot(input, self.W), axis=0)
            # initialize beta to zero
            self.beta = np.zeros((self.n_out, ))

        if self.gamma is None:
            #self.gamma = np.std(np.dot(input, self.W), axis=0)
            # initialize gamma to 1
            self.gamma = np.ones((self.n_out, ))

        # calculate intermidiate value
        self.x = np.dot(input, self.W)
        self.mean = np.mean(self.x, axis=0)
        self.var = np.var(self.x, axis=0)
        self.x_norm = (self.x - self.mean) / np.sqrt(self.var + 1e-8)

        # calculate linear output
        self.lin_output = self.gamma * self.x_norm + self.beta
        self.output = self.lin_output

        # RELU
        self.output[self.output<=0]=0
        return self.output

    def backward(self, delta):
        # d(Activation(lin_output))/d(lin_output)
        delta[self.lin_output <= 0] = 0

        # calculate the gradient of gamma and beta
        N, D = self.x.shape
        x_mean = self.x - self.mean
        std_inv = 1. / np.sqrt(self.var + 1e-8)

        dx_norm = delta * self.gamma
        dvar = np.sum(dx_norm * x_mean, axis=0) * -.5 * std_inv ** 3
        dmean = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2. * x_mean, axis=0)

        dxw = (dx_norm * std_inv) + (dvar * 2 * dmean / N) + (dmean / N)

        self.grad_gamma = np.sum(delta * self.x_norm, axis=0)
        self.grad_beta = np.sum(delta, axis=0)

        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(dxw))

        # calculate delta to pass
        delta_ = dxw.dot(self.W.T)

        # if not self.is_last:
        #     delta_ *= self.layer_mask
        # if not self.is_last:
        return delta_


    def dropout(self, input, rng=None):
        if rng is None:
            rng = np.random.RandomState(None)

        mask = rng.binomial(size=input.shape, n=1, p=1 - self.dropout_p)
        return mask

    def batch_norm(self, input, beta=np.array([0, 0])):
        gamma = np.ones([1, input.shape[0]])
        mean = np.mean(input)
        variance = np.mean((input - mean) ** 2, axis=0)
        input_hat = (input - mean) * 1.0 / np.sqrt(variance + self.epsl)
        out = gamma * input_hat + beta
        return out

    def update(self, my, lr):
        # momentum update
        self.v = my * self.v + lr * self.grad_W
        self.W -= self.v

        # normal update
        # self.W += -lr * self.grad_W
        self.gamma += -lr * self.grad_gamma
        self.beta += -lr * self.grad_beta


class MLP_bn:

    def __init__(self, layers, dropouts):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        ### initialize layers
        self.layers = []
        self.params = []
        self.epsilon = 1e-10
        self.dropout_masks = []
        self.dropouts = []

        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1],
                                           dropout=dropouts[i],
                                           is_last=(i == len(layers) - 2)))
    def forward(self, input):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            output = layer.forward(input)

            if i != len(self.layers) - 1 and layer.dropout_p != -1:
                mask = layer.dropout(output)
                output *= mask
                self.dropout_masks.append(mask)

            input = output
        return output

    def __softmax(self, x):

        exps = np.exp(x)
        return exps / (np.sum(exps, axis=1, keepdims=True))

    def cross_entropy(self, y, y_hat):
        reg = 1e-3

        probs = self.__softmax(y_hat)
        m = y_hat.shape[0]
        log_likelihood = -np.log(probs[range(m), y])
        loss = np.sum(log_likelihood) / m

        reg_loss = 0
        for l in self.layers:
            reg_loss += np.sum(l.W ** 2) * reg
        loss += reg_loss / len(self.layers)

        dscores = probs
        dscores[range(m), y] -= 1
        dscores /= m

        return loss, dscores

    def backward(self, delta):
        for i in reversed(range(len(self.layers))):
            delta = self.layers[i].backward(delta)
            if i != 0 and self.layers[i].dropout_p != -1:
                delta *= self.dropout_masks[i-1]

    def update(self, my, lr):
        for layer in self.layers:
            layer.update(my, lr)

    def iterate_minibatches(self, inputs, y, batchsize, shuffle=False):
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt, :], y[excerpt]

    def fit(self, X, y, data_val, y_val, learning_rate=0.01, my=0.9, epochs=100, batchsize=256):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """
        X = np.array(X)
        y = np.array(y)

        # the return loss result
        train_loss_return=list()
        train_acc_return=list()
        # the return acc result
        test_loss_return=list()
        test_acc_return=list()

        prev_time = time.time()
        for k in range(epochs):
            itr = 0
            #number of batch
            batch_num=X.shape[0]
            for batch in self.iterate_minibatches(X, y, batchsize=batchsize, shuffle=True):

                X_batch, y_batch = batch
                y_hat = self.forward(X_batch)
                #calculate loss and delta
                loss, delta = self.cross_entropy(y_batch, y_hat)
                self.backward(delta)
                self.update(my, learning_rate)

                #at the end of each epoch
                if itr % batch_num == 0:
                    # get the predict result
                    pred = np.argmax(y_hat, axis=1)
                    acc = np.mean(pred == y_batch)
                    # record training result
                    train_acc_return.append(acc)
                    train_loss_return.append(loss)
                    # print the result of the last batch at iteration
                    print("{}. loss: {}, accuracy:{}".format(k, loss, np.mean(pred == y_batch)))
                    # print the result of the testing
                    tloss, tacc=self.predict(data_val, y_val)
                    # record testing result
                    test_loss_return.append(tloss)
                    test_acc_return.append(tacc)

                itr += 1

        return train_acc_return, train_loss_return, test_acc_return, test_loss_return



    def predict(self, input, y):
        score = self.forward(input)
        loss, delta = self.cross_entropy(y, score)
        #print(score)
        pred = np.argmax(score, axis=1)
        acc=np.mean(pred == y)
        print("testing loss: {}, testing accuracy: {}".format(loss, acc))
        return loss, acc