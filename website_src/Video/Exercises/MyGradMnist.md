---
jupyter:
  jupytext:
    notebook_metadata_filter: nbsphinx
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python [conda env:week2]
    language: python
    name: conda-env-week2-py
  nbsphinx:
    execute: never
---

## Classifying MNIST with Le-Net (MyGrad and MyNN)


In this notebook, we will be training a convolutional neural network (using the Le-Net design described in [this paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)) to classify hand-written digits. We will be using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains labeled images of hand-written digits from 0 to 9. The MNIST dataset has a training set of 60,000 images and a test set of 10,000 images. 

You should have downloaded the [DataSets repo](https://github.com/CogWorksBWSI/DataSets), installed it, and set it up using `python setup.py develop` within that directory. This provides you with the mnist dataset, and a function for loading it, which we will use below.

We will be replicating the famous "LeNet" CNN architecture, which was one of the first convolutional neural network designs. We will explain the architecture and operations used in convolutional neural nets throughout this notebook. 


```python
import numpy as np
import mygrad as mg
from mygrad import Tensor

from noggin import create_plot
import matplotlib.pyplot as plt

%matplotlib notebook
```


### MNIST Data Loading and preprocessing


First, we will load in our data using handy functions from the datasets repo. If you haven't already, download the data by calling `download_mnist()`


```python
from datasets import load_mnist, download_mnist
download_mnist()
```


```python
# loading in the dataset with train/test data/labels
x_train, y_train, x_test, y_test = load_mnist()
```


What is the shape and data-types of these arrays? What is the shape of each individual image? How many color-channels does each number have.




Let's plot some examples from the MNIST dataset below


```python
img_id = 5

fig, ax = plt.subplots()
ax.imshow(x_train[img_id, 0], cmap="gray")
ax.set_title(f"truth: {y_train[img_id]}");
```


We will want to turn these 28x28 images into 32x32 images, for the sake of compatibility with the convolutions that we want to do. We can simply pad two rows/columns of zeros to all sides of the images


```python
# zero-pad the images
x_train = np.pad(x_train, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="constant")
x_test = np.pad(x_test, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="constant")
```


The original images stored unsigned 8bit integers for their pixel values. We need to convert these to floating-point values. Let's convert the images (not the labels) 32-bit floats.
You can use the `.astype()` array method to do this, and specify either `np.float32` or `"float32"` in the method call.


```python
# <COGINST>
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
# </COGINST>
```


Finally, we need to normalize these images. With cifar-10, we shifted the images by the mean and divided by the standard deviation. Here, let's be a little laze and simply normalize the images so that their pixel values lie on $[0, 1]$


```python
# <COGINST>
x_train /=  255.
x_test /= 255.

print(x_test.shape)
# </COGINST>
```


Complete the following classification accuracy function.

```python
def accuracy(predictions, truth):
    """
    Returns the mean classification accuracy for a batch of predictions.
    
    Parameters
    ----------
    predictions : Union[numpy.ndarray, mg.Tensor], shape=(M, D)
        The scores for D classes, for a batch of M data points

    truth : numpy.ndarray, shape=(M,)
        The true labels for each datum in the batch: each label is an
        integer in [0, D)
    
    Returns
    -------
    float
        The fraction of predictions that indicated the correct class.
    """
    return np.mean(np.argmax(predictions, axis=1) == truth) # <COGLINE>
```

## The "LeNet" Architecture



In the convnet to classify MNIST images, we will construct a CNN with two convolutional layers each structured as: 

```
conv layer --> relu --> pooling layer
```

, followed by two dense layers with a relu between them. Thus our network is:

```
CONV -> RELU -> POOL -> CONV -> RELU -> POOL -> FLATTEN -> DENSE -> RELU -> DENSE -> SOFTMAX
```




### Layer Details

CONV-1: 20 filters, 5x5 filter size, stride-1

POOL-1: 2x2, stride-2

CONV-2: 10 filters, 5x5 filter size, stride-1

POOL-2: 2x2, stride-2

DENSE-3: 20 neurons

DENSE-4: size-???  # hint: what should the dimensionality of our output be?

<!-- #region -->
### Activations

We will be using the "Glorot Uniform" initialization scheme for all of our layers' weights (the biases will be 0, which is the default). If you would like to read more about how Xavier Glorot explains the rationalization behind these weight initializations, look here for [his paper written with Yoshua Bengio](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).

This initialization scheme takes an additional "gain parameter", which will be $\sqrt{2}$ for us. Use the following syntax for specifying this gain:

```python
from mygrad.nnet.initializers import glorot_uniform

gain = {'gain': np.sqrt(2)}

# E.g. initializing a dense layer with glorot-uniform initialization
# and a gain of root-2
dense(d1, d2, 
      weight_initializer=glorot_uniform, 
      weight_kwargs=gain)
```
<!-- #endregion -->

```python
from mynn.layers.conv import conv
from mynn.layers.dense import dense

from mygrad.nnet.initializers import glorot_uniform
from mygrad.nnet.activations import relu
from mygrad.nnet.layers import max_pool
from mygrad.nnet.losses import softmax_crossentropy
```

```python
# Define your `Model`-MyNN class for the architecture prescribed above.

class Model:
    ''' A simple convolutional neural network. '''
    def __init__(self, num_input_channels, f1, f2, d1, num_classes):
        """
        Parameters
        ----------
        num_input_channels : int
            The number of channels for a input datum
            
        f1 : int
            The number of filters in conv-layer 1
        
        f2 : int
            The number of filters in conv-layer 2

        d1 : int
            The number of neurons in dense-layer 1
        
        num_classes : int
            The number of classes predicted by the model.
        """
        # Initialize your two convolution layers and two dense layers each 
        # as class attributes using the functions imported from MyNN
        #
        # We will use `weight_initializer=glorot_uniform` for all 4 layers
        
        # Note that you will need to compute `input_size` for
        # dense layer 1 : the number of elements being produced by the preceding conv
        # layer
        # <COGINST>
        init_kwargs = {'gain': np.sqrt(2)}
        self.conv1 = conv(num_input_channels, f1, 5, 5, 
                          weight_initializer=glorot_uniform, 
                          weight_kwargs=init_kwargs)
        self.conv2 = conv(f1, f2, 5, 5 ,
                          weight_initializer=glorot_uniform, 
                          weight_kwargs=init_kwargs)
        self.dense1 = dense(f2 * 5 * 5, d1, 
                            weight_initializer=glorot_uniform, 
                            weight_kwargs=init_kwargs)
        self.dense2 = dense(d1, num_classes, 
                            weight_initializer=glorot_uniform, 
                            weight_kwargs=init_kwargs)
        # </COGINST>


    def __call__(self, x):
        ''' Defines a forward pass of the model.
        
        Parameters
        ----------
        x : numpy.ndarray, shape=(N, 1, 32, 32)
            The input data, where N is the number of images.
            
        Returns
        -------
        mygrad.Tensor, shape=(N, num_classes)
            The class scores for each of the N images.
        '''
        
        # Define the "forward pass" for this model based on the architecture detailed above.
        # Note that, to compute 
        # We know the new dimension given the formula: out_size = ((in_size - filter_size)/stride) + 1
    
        # <COGINST>
        x = relu(self.conv1(x))
        x = max_pool(x, (2, 2), 2)
        x = relu(self.conv2(x))
        x = max_pool(x, (2, 2), 2)
        x = relu(self.dense1(x.reshape(x.shape[0], -1)))
        return self.dense2(x)
        # </COGINST>

    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model. """
        # Create a list of every parameter contained in the 4 layers you wrote in your __init__ function
        # <COGINST>
        params = []
        for layer in (self.conv1, self.conv2, self.dense1, self.dense2):
            params += list(layer.parameters)
        return params
        # </COGINST>

```

<!-- #region -->
Initialize the SGD-optimizer. We will be adding a new feature to our update method, known as ["momentum"](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum). The following is a sensible configuration for the optimizer:

```python
SGD(<your model parameters>, learning_rate=0.01, momentum=0.9, weight_decay=5e-04)
```
<!-- #endregion -->

```python
# Import SGD and initialize it as described above
# Also initialize your model
# <COGINST>
from mynn.optimizers.sgd import SGD

model = Model(f1=20, f2=10, d1=20, num_input_channels=1, num_classes=10)
optim = SGD(model.parameters, learning_rate=0.01, momentum=0.9, weight_decay=5e-04)
# </COGINST>
```

```python
plotter, fig, ax = create_plot(["loss", "accuracy"])
```

Using a batch-size of 100, train your convolutional neural network. Try running through 1 epoch of your data (i.e. enough batches to have processed your entire training data set once) - this may take a while. Plot training-loss and training accuracy, via noggin, for each batch. After each epoch, measure the *test* accuracy of your model on the entire test set - do not perform backprop for this stage. You should find that your network gets excellent performance.

Reference the cifar-10 (solution) notebook for guidance on this.

```python
# <COGINST>
batch_size = 100

for epoch_cnt in range(1):
    idxs = np.arange(len(x_train))  # -> array([0, 1, ..., 9999])
    np.random.shuffle(idxs)  
    
    for batch_cnt in range(len(x_train)//batch_size):
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        batch = x_train[batch_indices]  # random batch of our training data

        # compute the predictions for this batch by calling on model
        prediction = model(batch)

        # compute the true (a.k.a desired) values for this batch: 
        truth = y_train[batch_indices]

        # compute the loss associated with our predictions(use softmax_cross_entropy)
        loss = softmax_crossentropy(prediction, truth)

        # back-propagate through your computational graph through your loss
        loss.backward()

        # execute gradient descent by calling step() of optim
        optim.step()
        
        # compute the accuracy between the prediction and the truth 
        acc = accuracy(prediction, truth)
        
        # set the training loss and accuracy
        plotter.set_train_batch({"loss" : loss.item(),
                                 "accuracy" : acc},
                                 batch_size=batch_size)
    
    # Here, we evaluate our model on batches of *testing* data
    # this will show us how good our model does on data that 
    # it has never encountered
    # Iterate over batches of *testing* data
    for batch_cnt in range(0, len(x_test)//batch_size):
        idxs = np.arange(len(x_test))
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        batch = x_test[batch_indices] 
        
        with mg.no_autodiff:
            # get your model's prediction on the test-batch
            prediction = model(batch)
            
            # get the truth values for that test-batch
            truth = y_test[batch_indices]
            
            # compute the test accuracy
            acc = accuracy(prediction, truth)
        
        # log the test-accuracy in noggin
        plotter.set_test_batch({"accuracy": acc}, batch_size=batch_size)
    
    plotter.set_train_epoch()
    plotter.set_test_epoch()
plotter.plot()
# </COGINST>
```


Referencing the matplotlib code at the top of the notebook, visualize some images and check your model's predictions for them.

Also, use your model and the truth data to find images that the model *fails* to get right - plot some of these fail cases.

```python
# <COGLINE>
```
