# pydd

_Dark dresses in python, accelerated with jax._

A dark dress is an intermediate/extreme mass-ratio compact object binary where
the heavy object is surrounded by a dense dark matter halo. This repository
contains code for computing the waveforms for dark dresses that approximate
those produced by [`HaloFeedback`](https://github.com/bradkav/HaloFeedback). You
can use the scripts to run nested sampling to derive posteriors for observations
of these systems assuming a dark dress or GR-in-vacuum waveform models, and
compute the Bayes factor between the two.

The code uses the [`jax`](https://github.com/google/jax/) package to accelerate
the calculations.

## Getting started

First install the dependencies:

- jax: `pip install jax jaxlib`.
- [jaxinterp2d](https://github.com/adam-coogan/jaxinterp2d/): clone the repo and
  install with `pip install .`.

Then install `pydd` by cloning this repo and installing with `pip install .`.
Add `-e` to make the install [editable](https://pip.pypa.io/en/stable/cli/pip_install/#install-editable).
