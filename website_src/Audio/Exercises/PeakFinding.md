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
    display_name: Python [conda env:.conda-week1]
    language: python
    name: conda-env-.conda-week1-py
  nbsphinx:
    execute: never
---

<!-- #raw -->
.. meta::
    :description: Topic: matching audio, Category: Exercises
    :keywords: fingerprint, audio matching, local maxima
<!-- #endraw -->

# Exercises: Finding Local Peaks in a 2-D Array --

```python
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import iterate_structure

from typing import Tuple, Callable, List

%matplotlib notebook
```

## Toy Problem


We want to find the primary points of contact made by puppy-paws on a pressure-sensor.
There are $4$ images that are each $11\times14$ pixels.
Let's load and visualize this data.

```python
# loads four images of puppy paw print pressure data
paws = np.loadtxt("data/paws.txt").reshape(4, 11, 14)
print(paws.shape)
```

```python
# plots the paw prints
fig, ax = plt.subplots(nrows=2, ncols=2)
for n, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
    ax[i, j].imshow(paws[n])
```

For each "toe", we want to find the pixel with the maximum pressure.
This corresponds to a finding the local peaks in a 2-D image.
This is much more nuanced than finding the global maximum.
The term "local peak" is also not completely well defined - we need to specify what we mean by "local".


### Using SciPy's generate_binary_structure


We will use `scipy.ndimage.morphology.generate_binary_structure` to help us define the local neighborhood that we will consider when looking for 2-D peaks.

`generate_binary_structure` produces the "footprint" in which we look for neighbors.
This is simply a 2-D array of boolean values that indicate where we want to look within the footprint (i.e. `False` means ignore).
Using `generate_binary_structure(rank=2,connectivity=1)` means that, for a given pixel, we will check its two vertical and two horizontal neighbors when checking for the local maximum, aka, the "local peak".

Let's generate and visualize this specific footprint

```python
generate_binary_structure(rank=2, connectivity=1)
```

```python
fig, ax = plt.subplots()
ax.imshow(generate_binary_structure(rank=2, connectivity=1))
ax.set_title("Rank-2, Connectivity-1\nNeighborhood")
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([]);
```

(1.8.1) What is the "footprint" produced by `generate_binary_structure(rank=2,connectivity=2)`?
(The plot may be misleading, try printing out the array)

Boolean arrays behave like a binary mask when multiplied with a numerical array.
Try multiplying the rank-$2$, connectivity-$1$ binary structure (which is a 2-D array of booleans) by $2$.
Try to predict what the result will be before running your code.

```python
2 * generate_binary_structure(2, 1) # <COGLINE>
```

<!-- #region -->
What if we want to use a larger footprint? We can make use of `scipy.ndimage.morphology.iterate_structure`.
This allows us to set roughly the number of nearest neighbors (along a given direction) that that we want to included in the footprint.

For instance:
```python
>>> fp = generate_binary_structure(2,1)
>>> iterate_structure(fp, 2)
array([[False, False,  True, False, False],
       [False,  True,  True,  True, False],
       [ True,  True,  True,  True,  True],
       [False,  True,  True,  True, False],
       [False, False,  True, False, False]], dtype=bool)
```
<!-- #endregion -->

```python
fig, ax = plt.subplots()
fp = iterate_structure(generate_binary_structure(2, 1), 2)
ax.imshow(fp)
ax.set_title("Iterated Neighborhood (nearest neighbor=2)")
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([]);
```

### Finding the Actual Peaks

The following code slides this "local neighborhood mask" over our grid of 2D data (e.g. our spectrogram of amplitudes).
For whichever element the neighborhood is centered on, we see if:

- That center element is larger than some minimum threshold, which must be exceeded as a requirement to be considered "a peak"
- No neighbor (as defined by the neighborhood) is larger than that center element

If these conditions are true, then that center element is considered to be a local peak.
We then iterate to the next element in the 2D array and repeat the process; ultimately we will have iterated over the entire 2D array of data to so identify all of the local peaks.
Note that this is a relatively simple way of doing local peak-finding, and is certainly not the most optimal algorithm to do so.

Doing for-loops over large numpy arrays is typically something that we avoid doing due to considerations of speed.
But we do not have access to a vectorized peak-finding algorithm, so for-loops are what we have to stick with.
Fortunately, we can leverage a package called Numba to help speed up this code.
Numba provides a "just in time" (JIT) compiler that is able to translate (some aspects of) Python code into optimized machine code.
That is, whereas we have typically avoided writing for-loops over large arrays of data in Python in favor of vectorization, Numba enables us to write plain Python code using for-loops, but obtain a function that will run quickly, as if it had been implemented in a fast, compiled language like C.

Study the following code to understand what is going on.

```python
from numba import njit

# `@njit` "decorates" the `_peaks` function. This tells Numba to
# compile this function using the "low level virtual machine" (LLVM)
# compiler. The resulting object is a Python function that, when called,
# executes optimized machine code instead of the Python code
#
# The code used in _peaks adheres strictly to the subset of Python and
# NumPy that is supported by Numba's jit. This is a requirement in order
# for Numba to know how to compile this function to more efficient
# instructions for the machine to execute
@njit
def _peaks(
    data_2d: np.ndarray, rows: np.ndarray, cols: np.ndarray, amp_min: float
) -> List[Tuple[int, int]]:
    """
    A Numba-optimized 2-D peak-finding algorithm.

    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected.

    rows : numpy.ndarray, shape-(N,)
        The 0-centered row indices of the local neighborhood mask

    cols : numpy.ndarray, shape-(N,)
        The 0-centered column indices of the local neighborhood mask

    amp_min : float
        All amplitudes at and below this value are excluded from being local
        peaks.

    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location.
    """
    peaks = []  # stores the (row, col) locations of all the local peaks

    # Iterate over the 2-D data in col-major order
    # we want to see if there is a local peak located at
    # row=r, col=c
    for c, r in np.ndindex(*data_2d.shape[::-1]):
        if data_2d[r, c] <= amp_min:
            # The amplitude falls beneath the minimum threshold
            # thus this can't be a peak.
            continue

        # Iterating over the neighborhood centered on (r, c)
        # dr: displacement from r
        # dc: discplacement from c
        for dr, dc in zip(rows, cols):
            if dr == 0 and dc == 0:
                # This would compare (r, c) with itself.. skip!
                continue

            if not (0 <= r + dr < data_2d.shape[0]):
                # neighbor falls outside of boundary
                continue

            # mirror over array boundary
            if not (0 <= c + dc < data_2d.shape[1]):
                # neighbor falls outside of boundary
                continue

            if data_2d[r, c] < data_2d[r + dr, c + dc]:
                # One of the amplitudes within the neighborhood
                # is larger, thus data_2d[r, c] cannot be a peak
                break
        else:
            # if we did not break from the for-loop then (r, c) is a peak
            peaks.append((r, c))
    return peaks

# `local_peak_locations` is responsible for taking in the boolean mask `neighborhood`
# and converting it to a form that can be used by `_peaks`. This "outer" code is
# not compatible with Numba which is why we end up using two functions:
# `local_peak_locations` does some initial pre-processing that is not compatible with
# Numba, and then it calls `_peaks` which contains all of the jit-compatible code
def local_peak_locations(data_2d: np.ndarray, neighborhood: np.ndarray, amp_min: float):
    """
    Defines a local neighborhood and finds the local peaks
    in the spectrogram, which must be larger than the specified `amp_min`.

    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected

    neighborhood : numpy.ndarray, shape-(h, w)
        A boolean mask indicating the "neighborhood" in which each
        datum will be assessed to determine whether or not it is
        a local peak. h and w must be odd-valued numbers

    amp_min : float
        All amplitudes at and below this value are excluded from being local
        peaks.

    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location.

    Notes
    -----
    Neighborhoods that overlap with the boundary are mirrored across the boundary.

    The local peaks are returned in column-major order.
    """
    rows, cols = np.where(neighborhood)
    assert neighborhood.shape[0] % 2 == 1
    assert neighborhood.shape[1] % 2 == 1

    # center neighborhood indices around center of neighborhood
    rows -= neighborhood.shape[0] // 2
    cols -= neighborhood.shape[1] // 2

    return _peaks(data_2d, rows, cols, amp_min=amp_min)
```

Complete the following function.

```python
def local_peaks_mask(data: np.ndarray, cutoff: float) -> np.ndarray:
    """Find local peaks in a 2D array of data.

    Parameters
    ----------
    data : numpy.ndarray, shape-(H, W)

    cutoff : float
         A threshold value that distinguishes background from foreground

    Returns
    -------
    Binary indicator, of the same shape as `data`. The value of
    1 indicates a local peak."""
    # Generate a rank-2, connectivity-2 binary mask
    neighborhood_mask = generate_binary_structure(2, 2)  # <COGLINE>

    # Use that neighborhood to find the local peaks in `data`.
    # Pass `cutoff` as `amp_min` to `local_peak_locations`.
    peak_locations = local_peak_locations(data, neighborhood_mask, cutoff)  # <COGLINE>

    # Turns the list of (row, col) peak locations into a shape-(N_peak, 2) array
    # Save the result to the variable `peak_locations`
    peak_locations = np.array(peak_locations)

    # create a mask of zeros with the same shape as `data`
    mask = np.zeros(data.shape, dtype=bool)

    # populate the local peaks with `1`
    mask[peak_locations[:, 0], peak_locations[:, 1]] = 1
    return mask
```

Here is a function that will plot the paw prints next to the binary indicator of the local peaks.

```python
def plot_compare(
    data: np.ndarray,
    peak_finding_function: Callable[[np.ndarray], np.ndarray],
    cutoff: float = -np.inf,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the original data side-by-side with the binary indicator
    for the local peaks.

    Parameters
    ----------
    data : numpy.ndarray, shape=(N, H, W)
        N 2D arrays of shape (H, W)

    peak_finding_function : Callable[[ndarray], ndarray]
        Returns local peak indicator for 2D array

    cutoff : float, optional (default=-np.inf)
         A threshold value that distinguishes background from foreground

    Returns
    -------
    Tuple[matplotlib.Figure, matplotlib.Axes]
        The figure and axes objects of the plot
    """
    fig, ax = plt.subplots(nrows=len(data), ncols=2)
    for i, dat in enumerate(data):
        ax[i, 0].imshow(dat)
        ax[i, 1].imshow(peak_finding_function(dat, cutoff=cutoff))
    return fig, ax
```

(1.8.5) Now plot a comparison to assess how well our peak-finding function works.

```python
plot_compare(paws, local_peaks_mask); # <COGLINE>
```

What do you see in these right-column images?
Are these precisely the results we are looking for?
What seems to be off?

> <COGINST>1.8.5 Solution: No, the regions of the image that are blanketed in background (== 0) are also considered to be local peaks, since there are no smaller or larger points in their vicinity.
We need to subtract out the background from the image.
We can do this by finding `foreground = (data > 0)`, and require that the returned values are local peaks *and* are located in the foreground.</COGINST>

Inspect the paw print data.
What value is used to represent the background of the image?
What is the default value for `cutoff` in `plot_compare` for distinguishing between foreground and background?
Try adjusting this value in order to exclude the background from the peak-finding algorithm.


```python
plot_compare(paws, local_peaks_mask, cutoff=0.0); # <COGLINE>
```

Success! We are now finding local peaks in 2-D data!

To summarize this process, we:

 - Determined a neighborhood that was appropriate for measuring local peaks.
 - Created a max-filtered version of our data.
 - Demanded that our local peaks be in the "foreground" of our data.

This will be very useful to help us find the "fingerprint features" of a song, given its spectrogram (frequency vs time) data.


## Identifying "Foreground" vs "Background" in Real Data

Although this puppy paw print data set is pretty adorable, the fact that the paw print features are neatly embedded in a background of $0$s is too convenient.
In reality, we will likely face data where distinguishing background from a salient foreground is subtle (or perhaps entirely ill-posed).

Let's consider, for instance, the spectrogram data for the trumpet waveform.

```python
# running this cell loads the PCM-encoded data for the trumpet clip
import librosa

trumpet_audio, sampling_rate = librosa.load("data/trumpet.wav", sr=44100, mono=True)
```

```python
# using matplotlib's built-in spectrogram function
fig, ax = plt.subplots()

S, freqs, times, im = ax.specgram(
    trumpet_audio,
    NFFT=4096,
    Fs=sampling_rate,
    window=mlab.window_hanning,
    noverlap=4096 // 2,
)
fig.colorbar(im)

ax.set_xlabel("Time (sec)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Spectrogram of Audio Recording")
ax.set_ylim(0, 6000);
```

To help us identify a "foreground" in the log-amplitudes of the spectrogram, we will plot the *cumulative distribution* of the log-amplitudes.
This will allow us to identify a useful percentile below which we can consider all amplitudes to be "background".

The following function can be used to compute [an empirical cumulative distribution function](https://en.wikipedia.org/wiki/Empirical_distribution_function) (ECDF) of our data.

```python
import numpy as np

def ecdf(data):
    """Returns (x) the sorted data and (y) the empirical cumulative-proportion
    of each datum.

    Parameters
    ----------
    data : numpy.ndarray, size-N

    Returns
    -------
    Tuple[numpy.ndarray shape-(N,), numpy.ndarray shape-(N,)]
        Sorted data, empirical CDF values"""
    data = np.asarray(data).ravel()  # flattens the data
    y = np.linspace(1 / len(data), 1, len(data))  # stores the cumulative proportion associated with each sorted datum
    x = np.sort(data)
    return x, y
```

Let's get a feel for what `ecdf` does by using it to plot the cumulative distribution of our log-scaled spectrogram amplitudes.

```python
fig, ax = plt.subplots()

x, y = ecdf(np.log(S))
ax.plot(x, y)

ax.set_xlabel(r"$\log(|a_{k}|)$")
ax.set_ylabel(r"Cumulative proportion")
ax.set_title("Cumulative distribution of log-amplitudes")
ax.grid(True)
```

This cumulative distribution permits us to look up the percentiles of the log-amplitudes.
For example, we can find the log-amplitude below which $80\%$ of all the other present log-amplitudes fall (roughly $-2.9$).
According to the plot above, we see that roughly $90\%$ of all the log-amplitudes in our spectrogram fall beneath the value $0$.

**Consulting the shape of this cumulative distribution can help us distinguish a sensible threshold value to distinguish foreground and background**.
Here we see an "elbow" in the distribution just beyond the $60^\text{th}$ percentile.
We can identify the amplitude associated with this percentile with ease: just sort the amplitude data and extract the value at the integer index closest to `len(data) * 0.6`.


Let's find the log-amplitude associated with the $90\%$ percentile.
Read the documentation for `numpy.partition`, this function will enable us to rapidly find the amplitude associated with the desired percentile without having to sort all of our data.

```python
log_S = np.log(S).ravel()  # ravel flattens 2D spectrogram into a 1D array
ind = round(len(log_S) * 0.9)  # find the index associated with the 90th percentile log-amplitude
cutoff_log_amplitude = np.partition(log_S, ind)[ind]  # find the actual 90th percentile log-amplitude
cutoff_log_amplitude
```

We see that $90\%$ of all the log-amplitudes in the spectrogram fall below $-0.346$.
Thus $90\%$ of all of the Fourier coefficient amplitudes in this audio clip, $|a_{k}|$, fall beneath $e^{-0.346} \approx 0.71$.

We could use $-0.346$ as a cutoff value for distinguishing foreground from background when finding peaks in the log-amplitude spectrogram for our trumpet audio clip!
