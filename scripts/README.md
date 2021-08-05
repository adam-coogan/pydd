# Scripts

Use these to assess various observational prospects for dark dresses:

- Detectability: when can a dark dress' waveform be distinguished from noise?
  See `plot_snr.py`.
- Discoverability: how different do dark dress and GR-in-vacuum waveforms look?
  See `calc_vacuum_fits.py`, `run_ns.py`, `job_discoverability.sh` and `plot_vacuum_fits.py`.
- Measurability: how well could a dark dress' parameters be measured? See
  `run_ns.py`, `job_measurability.sh` and `plot_measurability.py`.

The final figures and data products are contained in subdirectories.

The scripts `fit_fb.py` and `plot_fb.py` are for calibrating and checking our
`f_b` scaling relation using
[`HaloFeedback`](https://github.com/bradkav/HaloFeedback) runs.
