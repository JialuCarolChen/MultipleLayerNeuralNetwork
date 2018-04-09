from data_loader import data_loader
from collections import Counter
from models.networks import MLP
import numpy as np

data = data_loader.DataLoader(False).load_data()
#
nn = MLP([128, 64, 32, 32, 10], dropouts=[0.3, 0.1, -1, -1], activation='relu')

# X=data.training_dev
X = (data.training_dev - np.min(data.training_dev)) / (np.max(data.training_dev) - np.min(data.training_dev))
loss = nn.fit(X, data.label_dev, data.training_val, data.label_val, my=0.95, learning_rate=1e-3, epochs=10000)
#
print("loss: {}".format(loss))