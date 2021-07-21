---
jupyter:
  jupytext:
    notebook_metadata_filter: nbsphinx
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python [conda env:.conda-week2]
    language: python
    name: conda-env-.conda-week2-py
  nbsphinx:
    execute: never
---

# Exercises: Exploring A Dataset --

A critical first step to tackling any data analysis problem is to familiarize ourselves with the the data that we are using.
It is all too common for tutorials in the world of "data science" to eagerly leap right into slick methods of analysis without spending even a moment to get a feel for the data being processed.
This can quickly betray the trusting reader, who may then apply the tutorial's methods to data that is not immediately amenable to the same means of analysis.
Here are some simple questions that we should ask about any data set:

- What is the format of our data?
   - What is the domain of values associated with your dataset? For example, if you are working with images whose pixels are stored as uint8 values (i.e. unsigned 8-bit integers), then each pixel will reside on the integer-valued domain $[0, 255]$ (a total of $2^8$ values).
   - What is the "dimensionality" of our data? That is, how many numbers or quantities are associated with each datum in your dataset?
   - Is all of our data saved in the same format?
   - Is this a "lossy" format? (E.g. is there a substantial amount noise introduced to our data due to things like compression?)

- How was our data collected?
   - What are the scales/units associated with our data?
   - Was all of the data collected under comparable circumstances?
   - What are the biases in/limitations to the data? (E.g. pictures of people scraped off of social media could be biased towards selfie-style pictures, where the faces are well-lit and are prominently framed with little clutter; there could also be prominent biases in the distributions of ages and ethnicities of people depicted in such data)
   - Are there measurement errors (either documented or apparent) associated with the data?

- What are the "statistics" of our data?
   - What are the minimum/maximum values of the various quantities in our data?
   - Can we visualize relationships between values in our data as a scatter or surface plot?
   - Can we visualize distributions of our data via things like histograms and [empirical cumulative distributions](https://en.wikipedia.org/wiki/Empirical_distribution_function)?

## Our Dataset: NBA Player Measurements
This notebook will step us through the process of exploring a simple dataset containing various measurements (e.g. height, weight, etc.) of NBA players who were part of the 2019 draft.
The data is saved using the [NetCDF-4 format](https://en.wikipedia.org/wiki/NetCDF), which is designed for storing scientific array data.
This data format does not affect our measurements in any important way - the values stored in the `nba_draft_measurements.nc` file reflect exactly the measurements that were collected.
As we will see, all of the length measurements in this dataset carry units of inches.
Furthermore, they were collected to the nearest quarter of an inch - this is a systematic source of error associated with the limited precision of the measurements that were made.

### Using the Xarray Library

We will load our data from the NetCDF-4 format using the powerful [xarray library](http://xarray.pydata.org/en/stable/index.html).
This is a Python package that allows one to work with multi-dimensional array data that has *labeled axes, units, and coordinates associated with it*.
Where bare NumPy arrays require us to relate crucial information about data, like measurement units and coordinates, via auxiliary documentation, xarray's data structures makes this information explicit and intimately associated with the array data.
Thus it enables us to process and manipulate our data while retaining critical context about each datum.

Once we have access to this data, our main goal will be to use the [matplotlib library](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Matplotlib.html) to visualize our data, and to glean interesting patterns from these visualizations.

Let's start by loading our data into an [xarray-Dataset](http://xarray.pydata.org/en/stable/data-structures.html#dataset)

```python
# run this cell
import matplotlib.pyplot as plt

%matplotlib notebook

import numpy as np

from pathlib import Path
import xarray as xr

# Here we use the xarray library to load the NetCDF-4 file into an xarray-Dataset
draft_data = xr.load_dataset(Path.cwd() / "data" / "nba_draft_measurements.nc")
```

Running the following cell will display the "repr" (representation) of this object.
Where most Python objects display strings as their reprs, a xarray object's repr is especially informative.
In a Jupyter notebook in should produce an interactive summary of its data, rendered via HTML elements.

Take some time to study the output of this cell.
What are the data variables (i.e. measurements) in this dataset?
What are the coordinates (i.e. identifiers) associated with these measured values?

```python
# Viewing the so-called "repr" of our dataset
draft_data
```

This data is an `xarray.Dataset` that stores measurements/info for 70 players from the 2019 NBA draft; these measurements are:

1. Height (no shoes) [inches]
2. Height (with shoes) [inches]
3. Weight [pounds]
4. Standing Reach [inches]
5. Wingspan [inches]
6. Body-fat Percentage
7. Hand-length [inches]
8. Hand-width [inches]
9. Player Position [PG: point guard, SG: shooting guard, C: center, PF: power-forward, SF: small forward]

Each of these measurements corresponds to a so-called **data variable** in the xarray-dataset; each of which is a shape-(70,) array.
Each entry across data variables corresponds to a specific player; thus the so-called **coordinates** that align these data are a shape-(70,) array of player names.


### Accessing the data


Each data variable can be accessed by metric name (you can use tab-completion to check the different metric names).
Evaluate `draft_data.height_no_shoes` in the cell below to access `shape-(70,)` array of player heights; note that the player names are still associated with these measurements.
This is an array-like object, thus it can be indexed into and sliced like a standard numpy array.

```python
# accessing a data variable from our dataset
# <COGINST>
draft_data.height_no_shoes
# </COGINST>
```

The underlying numpy array associated with a data variable can be accessed via the `.data` attribute.
Evaluate `draft_data.height_no_shoes.data` in the cell below to access the NumPy array that stores its data.

```python
# accessing the underlying numpy array of a data variable
# <COGINST>
draft_data.height_no_shoes.data
# </COGINST>
```

Data arrays support the same sort of elementwise (i.e. vectorized) operations as numpy arrays.
In the following cell, compute the height added to each player by his shoes.

```python
# Computing the height added to each player by his shoes
# <COGINST>
height_added = draft_data.height_with_shoes - draft_data.height_no_shoes
print(height_added)
# </COGINST>
```

Data arrays also have available to them the same sort of math methods, like `.sum()`.
Use `.mean()` and `.std()` to compute the average height added to a player by his shoes, along with the standard deviation.
What units are these values?

```python
# Computing the mean and standard deviation of the height added to each player by his shoes
# <COGINST>
height_added = draft_data.height_with_shoes - draft_data.height_no_shoes
print(f"mean height added: {height_added.mean()} [inches]"
      f"\nstd-dev: {height_added.std()} [inches]")
# </COGINST>
```

### Visualizing Our Dataset

The `xarray` library has [rich and convenient plotting utilities](http://xarray.pydata.org/en/stable/plotting.html).

For example, the following code will plot how a player's height and weight varies across player-position.
Do you notice any similarities or differences between the trends in these two plots?

```python
fig, axes = plt.subplots(nrows=2)
draft_data.plot.scatter(x="position", y="height_no_shoes", ax=axes[0])
draft_data.plot.scatter(x="position", y="weight", ax=axes[1])
axes[0].set_title("Metrics for NBA Draftees (2019)")
[ax.grid() for ax in axes];
```

The following plot shows display hand-width vs hand-height.

Do you notice anything about how the data falls along distinct vertical and horizontal lines in the plot?
What conclusion might you draw about the data-acquisition process to explain this "quantization" effect?

```python
fig, ax = plt.subplots()
draft_data.plot.scatter(x="hand_length", y="hand_width")
ax.grid()
```

Provide a guess as to why the data points seem to "snap to" an evenly spaced grid in the plot.

<COGINST>
ANSWER: The quantization effect shown above indicates that these measurements were recorded to the nearest quarter inch
</COGINST>


Plot the same sort of scatter plot, but for wingspan vs height (without shoes)

```python
# <COGINST>
fig, ax = plt.subplots()
draft_data.plot.scatter(x="height_no_shoes", y="wingspan")
ax.grid()
ax.set_xlabel("Height (without shoes) [inches]")
ax.set_ylabel("Wingspan [inches]");
ax.set_title("Wingspan vs Height for NBA Draftees from 2019");
# </COGINST>
```

Refer to the reading comprehension question ["Ordinary Least Squares in Python"](https://rsokl.github.io/CogWeb/Video/Linear_Regression.html#Linear-Least-Squares:-A-Closed-Form-Solution) from the previous section of CogWeb and complete the following function.
Check your answer against the solution that is provided at the end of the page.

```python
import numpy as np

def ordinary_least_squares(x, y):
    """
    Computes the slope and y-intercept for the line that minimizes
    the sum of squared residuals of mx + b and y, for the observed data
    (x, y).

    Parameters
    ----------
    x : numpy.ndarray, shape-(N,)
        The independent data. At least two distinct pieces of data
        are required.

    y : numpy.ndarray, shape-(N,)
        The dependent data in correspondence with ``x``.

    Returns
    -------
    (m, b) : Tuple[float, float]
        The optimal values for the slope and y-intercept
    """
    # <COGINST>
    N = x.size
    m = (np.matmul(x, y) - x.sum() * y.sum() / N) / (np.matmul(x, x) - (1 / N) * x.sum() ** 2)
    b = y.mean() - m * x.mean()
    return m, b
    # </COGINST>
```

Use the function `ordinary_least_squares` to compute the ideal slope and y-intercept for the line of least squares associated with the wingspan versus height (without shoes) data.
Next, plot this data along with the "best fit" linear model.
Use `ax.scatter(x, y)` to plot the original data and `ax.plot` to draw the model line;
you will want to specify a distinct color for your linear model.
Label your axes.

```python
# Plot wingspan vs height (no shoes)
# Plot m* x + b*


fig, ax = plt.subplots()

# plot the data using a scatter plot
draft_data.plot.scatter(x="height_no_shoes", y="wingspan")  # <COGLINE>

# compute the slope (m) and y-intercept (b) of
# the best-fit line
# <COGINST>
wingspan = draft_data.wingspan.data
height = draft_data.height_no_shoes.data
m, b = ordinary_least_squares(height, wingspan)
# </COGINST>

# create a domain of x-values between the minimum and maximum heights
# from the dataset (consider using `numpy.linspace`)
x = np.linspace(height.min(), height.max(), 1000) # <COGLINE>

# plot the best-fit line using the x-values
ax.plot(x, m * x + b, c="red", label="Least squares linear fit") # <COGLINE>

ax.grid(True)
ax.set_xlabel("Height [inches]")
ax.set_ylabel("Wingspan [inches]")
ax.set_title("Wingspan vs Height for NBA Draftees from 2019 ")
ax.legend();
```
