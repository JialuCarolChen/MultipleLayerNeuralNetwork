from data_loader import data_loader
from collections import Counter
from models.networks import MLP
from models.networksBN import MLP_bn
import matplotlib as plt
import numpy as np

data = data_loader.DataLoader(False).load_data()
#


# X=data.training_dev
X = (data.training_dev - np.min(data.training_dev)) / (np.max(data.training_dev) - np.min(data.training_dev))
# 0 center the trainning data
X = X - np.mean(X)
# another way of input pre-processing:



#X=(data.training_dev - np.mean(data.training_dev))/np.var(data.training_dev)
#print(X)
#nn = MLP([128, 64, 32, 32, 10], dropouts=[0.3, 0.1, -1, -1], activation='relu')
#loss = nn.fit(X, data.label_dev, data.training_val, data.label_val, my=0.95, learning_rate=0.001, epochs=1000)
#print("loss: {}".format(loss))


#batchNor
nn_bn = MLP_bn([128, 64, 10], dropouts=[0.5, -1], activation='relu')
loss_bn = nn_bn.fit(X, data.label_dev, data.training_val, data.label_val, my=0.95, learning_rate=0.01, epochs=1000)
print("loss: {}".format(loss_bn))