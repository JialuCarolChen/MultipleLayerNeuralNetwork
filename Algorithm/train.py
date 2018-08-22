from data_loader import data_loader
from collections import Counter
from models.networks import MLP
from models.networksBN import MLP_bn
import matplotlib as plt
import numpy as np


#load the data
dl = data_loader.DataLoader(False)
dl.split_train()
#encode the label

#print(dl.label[0:5])
#print(dl.label_dev[0:5])


### Data Preprocessing
## training data
# Substracting the mean
X = dl.training_dev - np.mean(dl.training_dev, axis=0)
# record training mean
#train_mean = np.mean(dl.training_dev)
#train_std = np.std(dl.training_dev)
#X = X - np.min(X, axis=0)/np.max(X, axis=0)-np.min(X, axis=0)
X /= np.std(X, axis=0)
##validation data
X_val = dl.training_val - np.mean(dl.training_val, axis=0)
X_val /= np.std(X_val, axis=0)

#print(X)
nn = MLP([128, 64, 32, 10], dropouts=[0.1, 0.1, -1], activation='relu')
loss = nn.fit(X, dl.label_dev, X_val, dl.label_val, my=0.95, learning_rate=1e-4, epochs=2000)
print("loss: {}".format(loss))


#batchNor
#nn_bn = MLP_bn([128, 32, 32, 32, 32, 10], dropouts=[0.5, 0.5, 0.5, 0.5, 0.5], activation='relu')
#loss_bn = nn_bn.fit(X, data.label_dev, data.training_val, data.label_val, my=0.95, learning_rate=0.1, epochs=5000)
#print("loss: {}".format(loss_bn))




