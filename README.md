
### Implementing Multi-Layer Neural Network from scratch with python

This project implements a Multilayer Perceptron Neural Network to solve a multi-class image classification. The structure and the size of the networks can be changed based on requirements. A selection of activation functions, weight initialization methods, and regularization modules like drop out and batch normalisation can also be specified.

### Instructions on running the code

To train a Multi-Layer Neural Network, place the input data (in .h5 format) under the root folder follow this structure

```
/Input
    /train.h5
     test.h5
     train_label.h5
```

To modify the setting of the Multiple Layer Neural Network, change the following parameters setting in the train.py
```python
nn = MLP(layers=[128, 512, 64, 10], dropouts=[0.5, 0.1, -1], activation='relu', norm="bn", update_type="momentum")
``` 
*	The layers parameter specify the structure of the networks. It is a list of the number of units for each layer.
*	The dropouts parameter is a list of the dropout rate for each layer (except the output layer)
*	The activation parameter specifies the activation function to use. The available options are: 'relu', 'tanh', 'logistic', 'l_relu' (Leaky Relu) and None 
*	The norm parameter specifies the normalisation module. If norm = 'bn', MLP is run with batch normalisation; if norm = 'wn', MLP is run with weight normalisation; otherwise run without batch normalisation and weight normalisation
*	The update_type parameter specifies the method used to update the parameters. If update_type = "momentum", MLP uses momentum update; if update_type = "nes_momentum", MLP uses nesterov momentum update; if update_type = None, just update with a small fraction of the gradients

To modify the setting of the training, change the parameters setting in the train.py

```python
train_acc, train_loss, test_acc, test_loss = nn.fit(X_train, data.label_dev, X_val, data.label_val, my=0.95,
                                                    learning_rate=1e-4,
                                                    epochs=max_epoch, batchsize=batch_size)
``` 
*	The my parameter is an additional parameter for momentum update
*	The learning_rate is the step size of updating the parameters
*	The epochs is the number of epochs to run 
*	The batchsize is the size of the mini-batch

Figures to comapre training/testing accuracy and loss change through training iterations will be output under the root folder. The predicted labels of the test data will be output as /Output/Predicted_labels.h5



