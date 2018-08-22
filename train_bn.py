from src.data_loader import data_loader
from collections import Counter
from src.models.networks import MLP
import matplotlib.pyplot as plt
import numpy as np
from src.models import networksBN



def data_prepare():
    # load and split the data to training and testing
    dl = data_loader.DataLoader(False)

    # data preprocessing
    ##training data
    X = dl.training_dev - np.mean(dl.training_dev, axis=0)
    X /= np.std(X, axis=0)

    ##validation data
    X_val = dl.training_val - np.mean(dl.training_val, axis=0)
    X_val /= np.std(X_val, axis=0)

    return X, dl.label_dev, X_val, dl.label_val


def train(layers, dropouts, my, lr, epoch, batchsize):
    # prepare the data
    train_data, train_label, val_data, val_label=data_prepare()
    # train and test
    nn = MLP(layers, dropouts)
    train_acc, train_loss, test_acc, test_loss = nn.fit(train_data, train_label, val_data,
                                                        val_label, my=my,
                                                        learning_rate=lr, epochs=epoch, batchsize=batchsize)
    #print(len(train_loss))
    #num of batch:
    #num_of_batch=50000/batchsize
    x_index_test=np.arange(1.0, epoch+1)
    #x_index_train=np.arange(0.0, epoch, 1.0 / num_of_batch)
    print(len(test_loss), len(x_index_test))
    # ploting the loss
    plt.figure(1)
    plt.plot(x_index_test, train_loss)
    plt.plot(x_index_test, test_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Loss of Training and Testing')
    plt.legend(['training', 'testing'], loc='upper left')
    plt.savefig('loss.png')
    # ploting the training
    plt.figure(2)
    plt.plot(x_index_test, train_acc)
    plt.plot(x_index_test, test_acc)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Accuracy of Training and Testing')
    plt.legend(['training', 'testing'], loc='upper left')
    plt.savefig('acc.png')
    return train_acc, train_loss, test_acc, test_loss



# A plotting function for comparison
def plot(index_xs, ys, xlabel, ylabel, title, legend, legend_pos, fig_path):
    for i in range(len(index_xs)):
        plt.plot(index_xs[i], ys[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend, loc=legend_pos)
    plt.savefig(fig_path)

# train a nerual network
#train_acc, train_loss, test_acc, test_loss= train(layers=[128, 64, 32, 10], dropouts=[0.1, 0.1, -1], my=0.95, lr=1e-2, epoch=600, batchsize=256)


def trainBN(layers, dropouts, my, lr, epoch, batchsize):
    # prepare the data
    train_data, train_label, val_data, val_label = data_prepare()
    # train and test
    nn_bn = networksBN.MLP_bn(layers, dropouts)
    train_acc, train_loss, test_acc, test_loss = nn_bn.fit(train_data, train_label, val_data,
                                                        val_label, my=my,
                                                        learning_rate=lr, epochs=epoch, batchsize=batchsize)
    # print(len(train_loss))
    # num of batch:
    # num_of_batch=50000/batchsize
    x_index_test = np.arange(1.0, epoch + 1)
    # x_index_train=np.arange(0.0, epoch, 1.0 / num_of_batch)
    print(len(test_loss), len(x_index_test))
    # ploting the loss
    plt.figure(1)
    plt.plot(x_index_test, train_loss)
    plt.plot(x_index_test, test_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Loss of Training and Testing (Batch Normalisation)')
    plt.legend(['training', 'testing'], loc='upper left')
    plt.savefig('loss_bn.png')
    # ploting the training
    plt.figure(2)
    plt.plot(x_index_test, train_acc)
    plt.plot(x_index_test, test_acc)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Accuracy of Training and Testing')
    plt.legend(['training', 'testing'], loc='upper left')
    plt.savefig('acc_bn.png')
    return train_acc, train_loss, test_acc, test_loss

# train a batch normalisation nerual network
train_acc_bn, train_loss_bn, test_acc_bn, test_loss_bn= trainBN(layers=[128, 64, 32, 10], dropouts=[0.1, 0.1, -1], my=0.95, lr=1e-3, epoch=300, batchsize=256)






