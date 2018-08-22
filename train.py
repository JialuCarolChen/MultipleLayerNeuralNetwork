import csv
import h5py

from src.utils import data_loader
from src.models.networks import MLP
import numpy as np

from src.utils.plotting import Plot
import timeit

# Record the time when the code starts
t1 = timeit.default_timer()

# Load the data
data = data_loader.DataLoader(False).load_data()

# Pre-process the data
mean = np.mean(data.training_dev, axis=0)
std = np.std(data.training_dev, axis=0)

X_train = data.training_dev - mean / std
X_val = data.training_val - mean / std
X_test = data.testing - mean / std

# We set the batch size as 200 and max epoch as 100 (In practice, we may set it up to 300000)
batch_size = 200
max_epoch = 30

# Define the multi layer perceptron class, and fit it.
nn = MLP([128, 512, 64, 10], dropouts=[0.5, 0.1, -1], activation='relu', norm="bn", update_type="momentum")

train_acc, train_loss, test_acc, test_loss = nn.fit(X_train, data.label_dev, X_val, data.label_val, my=0.95,
                                                    learning_rate=1e-4,
                                                    epochs=max_epoch, batchsize=batch_size)

# Make the prediction for the test data and save the data in a .h5 file.
_, output = nn.predict(X_test)
with h5py.File('../Output/Predicted_labels.h5', 'w') as hf:
    hf.create_dataset("label",  data=output)

# Record the accuracies and losses during the training
prefix = 'final'
f = open('results_{}.csv'.format(prefix), "w")
field_names = ['train_acc', 'train_loss', 'test_acc', 'test_loss']
writer = csv.DictWriter(f, fieldnames=field_names)
writer.writeheader()
row = dict()

for i in range(len(train_acc)):
    row['train_acc'] = train_acc[i]
    row['train_loss'] = train_loss[i]
    row['test_acc'] = test_acc[i]
    row['test_loss'] = test_loss[i]
    writer.writerow(row)
f.close()

# Plot the curve
plt = Plot(max_epoch, train_acc, train_loss, test_acc, test_loss, prefix=prefix)
plt.plot()

# Record the time when the code ends and print out the running time
t2 = timeit.default_timer()
print("running time is: {}".format(t2 - t1))


