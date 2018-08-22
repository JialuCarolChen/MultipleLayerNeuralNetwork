from src.models.weight_init import WeightInit
import numpy as np
import time
import math

class HiddenLayer(object):

    def __init__(self, n_in, n_out, W=None, b=None, is_last=False,
                 activation='tanh', dropout=0.5):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.epsl = 1e-5
        self.input = None
        #self.activation = Activation(activation).f
        #self.activation_deriv = Activation(activation).f_deriv
        self.activation_name = activation
        self.dropout_p = dropout
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        self.W = WeightInit(method=W, n_in=n_in, n_out=n_out).get_weights()
        if activation == 'logistic':
            self.W *= 4

        # W = np.random.randn(n_in, n_out)
        # u, s, v = np.linalg.svd(W)
        # self.W = u
        #
        self.b = np.zeros((1, n_out))
        self.v = np.zeros((n_in, n_out))

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        self.is_last = is_last
        self.layer_mask = None

    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b
        #RELU
        lin_output[lin_output<=0]=0
        self.output = lin_output


        # if not self.is_last:
        #     mask = self.dropout(self.output, 0.6)
        #     self.output *= mask
        #     self.layer_mask = mask
        self.input = input
        return self.output

    def backward(self, delta):
        lin_output = np.dot(self.input, self.W) + self.b
        #dActivation(lin_output)/d(lin_output)
        delta[lin_output <= 0] = 0
        #calculate W gradient
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        #calculate the delta for next layer
        delta_ = delta.dot(self.W.T)
        self.grad_b = np.sum(delta, axis=0, keepdims=True)
        #self.grad_W += 1e-3 * self.W
        # return delta_ for next layer
        # delta_ = delta.dot(self.W.T) * self.activation_deriv(self.input)
        # delta_ = delta.dot(self.W.T)
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
        self.v = my * self.v + lr * self.grad_W
        self.W -= self.v

        # c

        self.b -= lr * self.grad_b


class MLP:

    def __init__(self, layers, dropouts, activation='tanh'):
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

        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation=activation,
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

    def fit(self, X, y, data_val, y_val, learning_rate=0.01, my=0.9, epochs=100, batchsize=500):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """
        X = np.array(X)
        y = np.array(y)

        to_return = np.zeros(epochs)
        acc_return = np.zeros(epochs)

        prev_time = time.time()
        for k in range(epochs):
            itr = 0
            for batch in self.iterate_minibatches(X, y, batchsize=batchsize, shuffle=True):

                X_batch, y_batch = batch


                y_hat = self.forward(X_batch)
                #calculate loss and delta
                loss, delta = self.cross_entropy(y_batch, y_hat)
                self.backward(delta)
                self.update(my, learning_rate)

                to_return[k] = np.mean(loss)

                if itr % 10 == 0:
                    pred = np.argmax(y_hat, axis=1)
                    print("{}. loss: {}, accuracy:{}".format(k, to_return[k], np.mean(pred == y_batch)))
                    self.predict(data_val, y_val)

                itr += 1

        return to_return


    def predict(self, input, y):
        score = self.forward(input)
        #print(score)
        pred = np.argmax(score, axis=1)
        print("testing accuracy: {}".format(np.mean(pred == y)))
