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
    display_name: Python [conda env:week3]
    language: python
    name: conda-env-week3-py
  nbsphinx:
    execute: never
---

```python
from collections import defaultdict

import numpy as np

from mynn.layers.dense import dense
from mynn.optimizers.adam import Adam

from mygrad.nnet.losses import softmax_crossentropy
from mygrad.nnet.initializers import glorot_normal
from mygrad.nnet.activations import relu

import mygrad as mg
import matplotlib.pyplot as plt

%matplotlib notebook
```

# Simple RNN Cell in MyGrad --

In this notebook, we will implement a simple RNN model that can be used for sequence classification problems.
We'll apply this RNN to the **classification problem of determining if a sequence of digits (0-9) is the concatentation of two identical halves.**

For example:
- `[1, 2, 3, 1, 2, 3]` -> contains identical halves
- `[1, 9, 2, 1, 8, 3]` -> does not contain identical halves

Our model will take a single sequence of data ($x$) of shape $(T, C)$, where $T$ is the length of our sequence and $C$ is the dimensionality of each entry in our sequence, and produce $K$ classification scores (assuming there are $K$ classes for the problem).

In the context of word-embeddings, if each word in our vocabulary has a 50-dimensional word-embedding representation, and we have with a sentence containing 8 words, then $x$ would have a shape $(8, 50$) - representing that sentence numerically. Our model would produce $K$ classification scores for this input data.

**The actual problem that we are solving is the following:**
> Given a sequence of digits, return 1  if the first half and second half of a sequence are identical and 0 otherwise.

We'll be using the following update equations for a simple RNN cell:
<br/>
<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $h_t = ReLU(x_t W_{xh} + h_{t-1} W_{hh} + b_h)$
<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $y_{T-1} = h_{T-1} W_{hy} + b_y$

where $h_t$ is the hidden (or recurrent) state of the cell and $x_t$ is the sequence-element at step-$t$, for $t=0, 1, \dots, T-1$. ($T$ is the length of our sequence.) $y_{T-1}$ is the output. The $W$ and $b$ parameters are the *learnable parameters of our model*. Specifically:

- $x_t$ is a descriptor-vector for entry-$t$ in our sequence of data. It has a shape-$(1, C)$.
- $h_t$ is a "hidden-descriptor", which encodes information about $x_t$ *and* information about the preceding entries in our sequence of data, via $h_{t-1}$. It has a shape-$(1, D)$, where $D$ is the dimensionality that we choose for our hidden descriptors (akin to layer size).
- $W_{xh}$ and $b_h$ hold dense-layer weights and biases, respectively, which are used to process our data $x_t$ in order to form $h_t$. Thus $W_{xh}$ has shape $(C, D)$ and $b_h$ has shape-$(1,D)$.
- $W_{hh}$ hold dense-layer weights, which are used to process our previous hidden-descriptor $h_{t-1}$ in order to form $h_t$. Thus $W_{hh}$ has shape $(D, D)$.
- $W_{hy}$ and $b_y$ hold dense-layer weights and biases, respectively, which are used to process our final hidden-descriptor $h_T$ in order to produce our classification scores, $y_T$. Thus $W_{hy}$ has shape $(D, K)$ and $b_h$ has shape-$(1,K)$. Where $K$ is our number of classes. See that, given our input sequence $x$, we are ultimately producing $y_{T-1}$ of shape-$(1, K)$.

The basic idea is to have the forward pass in the model iterate over all elements in the input sequence, applying the update equations at each step.

Then we'll compute the loss between the final output $y_{T-1}$ and the target classification, perform backpropagation through the computational graph to compute gradients (known as "backpropagation through time" or "BPTT" in RNNs), and update parameters using some form of gradient descent.



<!-- #region -->
## Define Recurrent Model Class

First create a recurrent model class using MyGrad and MyNN with the following properties:
* `__init__`
 * Takes three parameters: dim_input ($C$), dim_recurrent ($D$), dim_output ($K$)
 * Creates three dense layers (required for update equations)
  * Note: one of the dense layers doesn't need a bias since it would be redundant. You can specify `bias=False` when initializing your dense layer.
  * You can leave the bias out of the dense layer corresponding to $W_{hh}$
* `__call__`
 * Creates the initial hidden state ($h_{t=-1}$) as an array of zeros, shape-(1, D)
 * Iterates over the $T$-axis (rows) of the input sequence $x$ and computes the successive hidden states $h_{t=0}, h_{t=1} \cdots, h_{t={T-1}}$
 * After processing the all $T$ items in your sequence, computes/returns final output $y_{T-1}$
* `parameters`
 * Returns the tuple of all the learnable parameters in your model.


As usual, we will feed our `(1, K)` scores to softmax-crossentropy loss, thus there is no need for an activation function on $y_{T-1}$, since softmax is built in to the loss.

Use `glorot_normal` for your dense weight initializations.
<!-- #endregion -->

```python
class RNN():
    """Implements a simple-cell RNN that produces a single output at the
    end of the sequence of input data."""
    def __init__(self, dim_input, dim_recurrent, dim_output):
        """ Initializes all layers needed for RNN

        Parameters
        ----------
        dim_input: int
            Dimensionality of data passed to RNN (C)

        dim_recurrent: int
            Dimensionality of hidden state in RNN (D)

        dim_output: int
            Dimensionality of output of RNN (K)
        """
        # Initialize one dense layer for each matrix multiplication that appears
        # in the simple-cell RNN equation; name these "layers" in ways that make
        # their correspondence to the equation obvious
        # <COGINST>
        self.fc_x2h = dense(dim_input, dim_recurrent, weight_initializer=glorot_normal)
        self.fc_h2h = dense(dim_recurrent, dim_recurrent, weight_initializer=glorot_normal, bias=False)
        self.fc_h2y = dense(dim_recurrent, dim_output, weight_initializer=glorot_normal)
        # </COGINST>


    def __call__(self, x):
        """ Performs the full forward pass for the RNN.

        Note that we only care about the last y - the final classification scores for the full sequence.


        Parameters
        ----------
        x: Union[numpy.ndarray, mygrad.Tensor], shape=(T, C)
            The one-hot encodings for the sequence

        Returns
        -------
        mygrad.Tensor, shape=(1, K)
            The final classification scores, produced at the end of the sequence
        """
        # Initialize the hidden state h_{t=-1} as zeros
        #
        # You will want to loop over each x_t to compute the corresponding h_t.
        #
        # A standard for-loop is appropriate here. Be mindful of what the shape
        # of x_t should be versus the shape of the item that it produced by the
        # for-loop.
        #
        # Note that you can do a for-loop over a mygrad-tensor and it will
        # produce sub-tensors that are tracked by the computational graph.
        # I.e. mygrad will be able to still "backprop" through your for-loop!
        # <COGINST>
        h = np.zeros((1, self.fc_h2h.weight.shape[0]), dtype=np.float32)
        for x_t in x:
            h = relu(self.fc_x2h(x_t[np.newaxis]) + self.fc_h2h(h))

        return self.fc_h2y(h)
        # </COGINST>


    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.

        This can be accessed as an attribute, via `model.parameters`

        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model
        """
        return self.fc_x2h.parameters + self.fc_h2h.parameters + self.fc_h2y.parameters # <COGLINE>
```

<!-- #region -->
## Data Generation

We'll apply this new network to the **problem of determining if a sequence of digits (0-9) is the concatentation of two identical halves.**

For example:
- `[1, 2, 3, 1, 2, 3]` -> contains identical halves
- `[1, 9, 2, 1, 8, 3]` -> does not contain identical halves

We will be representing each digit using the so-called "**one-hot encoding**"
 * 0 $\longrightarrow$ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 * 1 $\longrightarrow$ [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 * 2 $\longrightarrow$ [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
 * 3 $\longrightarrow$ [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
 * $\vdots$
 * 9 $\longrightarrow$ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

Thus a sequence of $T$ one-hot encoded digits will be represented by a shape-$(T,C=10)$ array.

For example, the sequence
```python
# length-4 sequence
array([2, 0, 2, 0])
```
Would have the one-hot encoding
```python
# shape-(4, 10)
array([[ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
```

Create a function to generate a sample sequence that does the following:
* allows you to specify min and max pattern length
* randomly chooses a pattern length in the specified range
* randomly generates a sequence of integers (0 through 9) of that length
* sets first half of sequence equal to pattern
* randomly chooses whether first and second half of sequence should match or not (with probability 0.5)
* creates second half of sequence accordingly
* creates float32 numpy array `x` of shape (T, 10) where row i is one-hot encoding of item i in sequence
* creates int16 numpy array `y` of shape (1,) where `y = array([1])` if the patterns match and `array([0])` otherwise
* returns `(x, y, sequence)` (note that sequence is returned mainly just for debugging)

Note: `np.random.rand() < 0.5` returns `True` with 50% probability. This will come in handy!

For example, if you randomly generate the sequence [2, 0, 2, 0] (which has a pattern-length of 2, whose first half does match the second half, which should occur 50% of the time), the output of your function should be:
```python
# x: one-hot encoded version of the sequence, shape-(4,10)
array([[ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])     

# y: the halves of the sequence do match -> 1
array([ 1])

# sequence
array([2, 0, 2, 0])
```
<!-- #endregion -->

```python
def generate_sequence(pattern_length_min=1, pattern_length_max=10, palindrome=False):
    """
    Randomly generate a sequence consisting of two equal-length patterns of digits,
    concatenated end-to-end.

    There should be a 50% chance that the two patterns are *identical* and a 50%
    chance that the two patterns are distinct.

    Parameters
    ----------
    pattern_length_min : int, optional (default=1)
       The smallest permissable length of the pattern (half the length of the
       smallest sequence)

    pattern_length_max : int, optional (default=10)
       The longest permissable length of the pattern (half the length of the
       longest sequence)

    palindome : bool, optional (default=False)
        If `True`, instead of a sequence with the two identical patterns, generate
        a palindrome instead.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        1. the one-hot encoded sequence; shape-(T, 10)
        2. the label for the sequence: 0 (halves don't match), 1 (halves match); shape-(1,)
        3. the actual sequence of digits; shape-(T,)
    """
    # <COGINST>
    pattern_length = np.random.randint(pattern_length_min, pattern_length_max + 1)
    pattern = np.random.randint(0, 10, pattern_length)
    match = np.random.rand() >= 0.5

    sequence = np.zeros(2 * pattern_length, dtype=np.int64)
    sequence[:pattern_length] = pattern
    if match:
        sequence[pattern_length:] = pattern[::-1] if palindrome else pattern
    else:
        # non-matching second half
        second_half = np.random.randint(0, 10, pattern_length)
        while np.array_equal(second_half, pattern):
            second_half = np.random.randint(0, 10, pattern_length)
        sequence[pattern_length:] = second_half

    # one-hot encoding of digits 0 through 9
    # x = np.zeros(pattern_length * 2, 10, dtype)
    x = np.zeros((len(sequence), 10), dtype=np.float32)
    y = np.array([1.0 if match else 0], dtype=np.int16)

    for i, ch in enumerate(sequence):
        x[i, ch] = 1

    return x, y, sequence
    # </COGINST>
```

Test your `generate_sequence` function manually.
- Does it produce sequences within the desired length bounds?
- Does `x` correspond to `sequence`, with the appropriate one-hot encoding?
- Does `y` indicate `array([1])` when the halves of the sequence match?

Consider writing some code with assert statements that will raise if any of these checks fail/

```python
# <COGINST>
# these checks are optional

for i in range(100):
    x, y, seq = generate_sequence()
    assert len(x) == len(seq)
    assert np.all(seq[:len(seq)//2] == seq[-len(seq)//2:]).item() is bool(y.item())
    assert np.all(x[:len(x)//2] == x[-len(x)//2:]).item() is bool(y.item())
# </COGINST>
```

Set up a noggin plot, as you will want to observe the loss and accuracy during training.

```python
from noggin import create_plot
plotter, fig, ax = create_plot(["loss", "accuracy"]) # <COGLINE>
```

Recall that each digit has a one-hot encoding, which means that $C=10$ (`input_dim`). A sensible hidden-descriptor dimensionality is $D=50$ (`dim_recurrent`). Lastly, we are solving a *two-class* classification problem (0 $\rightarrow$ no pattern match, 1 $\rightarrow$ pattern match), and thus $K=2$ (`dim_output`). Initialize your model accordingly.

Set up an Adam optimizer. Pass the Adam optimizer your model's learnable parameters. Otherwise use its default learning rate and other hyperparameters. Feel free to mess with these later.



```python
# <COGINST>
model = RNN(dim_input=10, dim_recurrent=50, dim_output=2)
optimizer = Adam(model.parameters)
# </COGINST>
```

<!-- #region -->
Train the model for 100000 iterations. Instead of pre-generating a set of training sequences, we'll use a strategy of randomly sampling a new input sequence every iteration using the method you created earlier. Use pattern_length_min = 1 and pattern_length_max = 10.

**Do not plot batch-level metrics. We will be processing so many sequences, that plotting all the losses and accuracies will become a performance bottleneck**. You can set your loss and accuracy for each batch without plotting, using

```python
plotter.set_train_batch({"loss":loss.item(), "accuracy":acc},
                        batch_size=1,
                        plot=False)
```

And then for every 500th batch (or whatever you want), call:

```python
plotter.set_train_epoch()
```

This will plot mean statistics for your model's performance instead of the accuracy and loss for every single input.
<!-- #endregion -->

```python
# <COGINST>
plot_every = 500

for k in range(100000):
    x, target, sequence = generate_sequence(palindrome=False)

    output = model(x)

    loss = softmax_crossentropy(output, target)

    loss.backward()
    optimizer.step()

    acc = float(np.argmax(output.squeeze()) == target.item())

    plotter.set_train_batch({"loss":loss.item(), "accuracy":acc}, batch_size=1, plot=False)

    if k % plot_every == 0 and k > 0:
        plotter.set_train_epoch()
# </COGINST>
```

### Accuracy vs Sequence Length

Create a plot of accuracy vs sequence length. To do so, randomly generate sequences (which will be of various lengths), apply the trained model to get the predicted outputs, and record whether the model predictions are correct or not. Then compute accuracy for sequences of length 2, for sequences of length 4, etc. (hint: Keep track of total and total correct for each possible length).

MyGrad note: Because we are simply evaluating the model and have no reason to compute gradients, use the `no_autodiff` context manager to tell MyGrad not to keep track of the computational graph and speed up the evaluations.

```python
# <COGINST>
length_total = defaultdict(int)
length_correct = defaultdict(int)

with mg.no_autodiff:
    for i in range(100000):
        if i % 5000 == 0:
            print("i = %s" % i)

        x, target, sequence = generate_sequence()
        output = model(x).squeeze()

        length_total[len(sequence)] += 1
        if np.argmax(output) == target.item():
            length_correct[len(sequence)] += 1

fig, ax = plt.subplots()
x, y = [], []
for i in range(2, 20, 2):
    x.append(i)
    y.append(length_correct[i] / length_total[i])
ax.plot(x, y);
# </COGINST>
```

What do you notice about accuracy as sequence length increases? Is this expected? What might make long sequences hard to deal with? Discuss with a neighbor!

What happens if you apply the model to a sequence that's longer than examples it's been trained on? What happens if we train on and try to classify palindromes? Try messing around with our model and exploring the results.


### View the computational graph formed from processing a sequence

We can view the computational graph that results from feeding a sequence through the RNN using MyGrad's awesome `build_graph` capability.

```python
from mygrad.computational_graph import build_graph
x, target, sequence = generate_sequence()

output = model(x)

loss = softmax_crossentropy(output, target)
build_graph(loss, names=locals(), render=True)
```
