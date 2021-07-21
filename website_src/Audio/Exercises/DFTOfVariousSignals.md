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
    display_name: Python 3
    language: python
    name: python3
  nbsphinx:
    execute: never
---

<!-- #raw raw_mimetype="text/restructuredtext" -->
.. meta::
   :description: Topic: Discrete Fourier Transforms, Category: Exercises
   :keywords: applications, examples, dft, Fourier spectrum
<!-- #endraw -->

<div class="alert alert-warning">

**Source Material**:

The following exercises are adapted from Chapter 7 of [Mark Newman's book, "Computational Physics"](http://www-personal.umich.edu/~mejn/cp/).

</div>



# Exercises: DFTs of Various Signals

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

%matplotlib notebook
```

For all problems, take $N = 1,000$ as the number of samples

<!-- #region -->
(1.5.1) Perform a DFT on a single period of a square wave of amplitude $1$.
Its period is $5$ seconds long.
Plot the wave form versus time and plot the Fourier spectrum, $|a_{k}|$ vs $\nu_{k}$.

Here is some code that you can use to plot these side-by-side:

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
ax1.plot(t, y, marker="o")  # plot time (t) and waveform (y)
ax1.grid()
ax1.set_xlabel("t (seconds)")

ax2.stem(freqs, amps, basefmt=" ", use_line_collection=True) # plot frequency (freqs) and amplitudes (amps)
ax2.set_xlim(0, 10)
ax2.grid()
ax2.set_ylabel(r"$|a_{k}|$")
ax2.set_xlabel(r"$\nu_{k}$ (Hz)")
fig.tight_layout()
```

A square wave is like a sine-wave, except a square wave only takes on values of $1$ (wherever $\sin(x)$ is positive) or $-1$ (wherever $\sin(x)$ is negative).
<!-- #endregion -->

```python
# <COGINST>
def fourier_complex_to_real(
    complex_coeffs: np.ndarray, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts complex-valued Fourier coefficients (of
    real-valued data) to the associated amplitudes and
    phase-shifts of the real-valued sinusoids

    Parameters
    ----------
    complex_coeffs : numpy.ndarray, shape-(N//2 + 1,)
        The complex valued Fourier coefficients for k=0, 1, ...

    N : int
        The number of samples that the DFT was performed on.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        (amplitudes, phase-shifts)
        Two real-valued, shape-(N//2 + 1,) arrays
    """
    amplitudes = np.abs(complex_coeffs) / N

    # |a_k| = 2 |c_k| / N for all k except for
    # k=0 and k=N/2 (only if N is even)
    # where |a_k| = |c_k| / N
    amplitudes[1 : (-1 if N % 2 == 0 else None)] *= 2

    phases = np.arctan2(-complex_coeffs.imag, complex_coeffs.real)
    return amplitudes, phases


def square_filter(x):
    return 1 if x >= 0 else -1


T = 5
N = 1000
t = np.arange(N) / N * T
y = [square_filter(i) for i in np.sin(2 * np.pi * t / T)]

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
ax1.plot(t, y, marker="o")
ax1.grid()
ax1.set_xlabel("t (seconds)")

freqs = np.arange(len(y) // 2 + 1) / T
amps, phases = fourier_complex_to_real(np.fft.rfft(y), N=N)

ax2.stem(freqs, amps, basefmt=" ", use_line_collection=True)
ax2.set_xlim(0, 10)
ax2.grid()
ax2.set_ylabel(r"$|a_{k}|$")
ax2.set_xlabel(r"$\nu_{k}$ (Hz)")
fig.tight_layout()
# </COGINST>
```

(1.5.2) Perform a DFT for the simple linear function $y_{t} = t$ for $1000$ seconds in duration.
Plot both the waveform and the Fourier spectrum,  $|a_{k}|$ vs $\nu_{k}$.

```python
# <COGINST>
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

N = 1000
T = 1000
t = np.arange(N) / N * T
y = t

ax1.plot(y)
ax1.set_xlabel("t (s)")

freqs = np.arange(len(y) // 2 + 1) / T
amps, phases = fourier_complex_to_real(np.fft.rfft(y), N=len(y))

ax2.stem(freqs, amps, basefmt=" ", use_line_collection=True)
ax2.set_xlim(0, 0.11)
ax2.grid()
ax2.set_ylabel(r"$|a_{k}|$")
ax2.set_xlabel(r"$\nu_{k}$ (Hz)")
fig.tight_layout()
# </COGINST>
```


(1.5.3) Perform a DFT for the modulated wave $\sin\!\big(\frac{\pi t}{Q}\big) \sin\!\big(\frac{20 \pi t}{Q}\big)$, where $Q=5\:\mathrm{seconds}$.
Sample this signal **over two periods of the lower-frequency term (a.k.a the modulating term)**.

- What are the frequencies of the respective terms in our modulated wave?
- Do you expect that these two frequencies present in the Fourier spectrum? Why might we expect for different frequencies to be prominent in our series (hint: compare the functional form of this waveform to that of a Fourier series)?
- Use the relationship $\sin{(a)}\sin{(b)}=\frac{1}{2}(\cos{(a-b)} - \cos{(a+b)})$ to rewrite this modulated wave as a sum of cosines.
From this, predict the number, locations, and heights of the peaks in your Fourier spectrum.
- Plot the wave form vs time and plot the Fourier spectrum, $|a_{k}|$ vs $\nu_{k}$.
Be sure to zoom in on your peaks.

> <COGINST> 1.5.3 Solution:
>    
> - The frequencies of the terms in the waveform are $\frac{1}{10}\:\mathrm{Hz}$ and $2\:\mathrm{Hz}$, respectively
> - The modulated waveform consists of a **product** of sine-waves, whereas our Fourier series represents a **sum** of waves.
> We shouldn't expect that a product of waves should be reproduced by a sum of waves of the same frequencies.
> - By the given identity, $\sin\!\big(\frac{\pi t}{5}\big) \sin\big(\frac{20 \pi t}{5}\big)=\frac{1}{2}\big(\cos{\big(2\pi\frac{19}{10}t\big)} - \cos{\big(2\pi\frac{21}{10}t\big)}\big)$.
> Thus we should see two distinct peaks, one at $1.9\;\mathrm{Hz}$ and one at $2.1\;\mathrm{Hz}$.
> Both peaks should have a height of $0.5$. </COGINST>

```python
# <COGINST>
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

T = 2 * 5
N = 1000
t = np.arange(N) * T / N
y = np.sin(np.pi * t / 5) * np.sin(20 * np.pi * t / 5)
ax1.plot(t, y)
ax1.grid()
ax1.set_xlabel("t (seconds)")

freqs = np.arange(len(y) // 2 + 1) / T
amps, phases = fourier_complex_to_real(np.fft.rfft(y), N=len(y))

ax2.stem(freqs, amps, basefmt=" ", use_line_collection=True)
ax2.set_xlim(1, 3)
ax2.grid()
ax2.set_ylabel(r"$|a_{k}|$")
ax2.set_xlabel(r"$\nu_{k}$ (Hz)")
fig.tight_layout()
# </COGINST>
```

(1.5.4) Perform a DFT on a noisy (i.e. random) signal that is centered around $0$.
Have the noisy signal last for $5$ seconds.

```python
# <COGINST>
def noise(t: np.ndarray) -> np.ndarray:
    return np.random.rand(*t.shape) - 0.5


T = 5
N = 1000
t = np.arange(N) / N * T
y = noise(t)
freqs = np.arange(len(y) // 2 + 1) / T
amps, phases = fourier_complex_to_real(np.fft.rfft(y), N=len(y))

fig, ax = plt.subplots()
ax.stem(freqs, amps, basefmt=" ", use_line_collection=True)
ax.grid()
ax.set_ylabel(r"$|a_{k}|$")
ax.set_xlabel(r"$\nu_{k}$ (Hz)")
fig.tight_layout()
# </COGINST>
```
