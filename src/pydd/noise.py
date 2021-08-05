"""
Noise PSDs.
"""

def S_n_LISA(f):
    """
    LISA noise PSD, averaged over sky position and polarization angle.

    Reference:
        Travis Robson et al 2019 Class. Quantum Grav. 36 105011
        https://arxiv.org/abs/1803.01944
    """
    return (
        1
        / f ** 14
        * 1.80654e-17
        * (0.000606151 + f ** 2)
        * (3.6864e-76 + 3.6e-37 * f ** 8 + 2.25e-30 * f ** 10 + 8.66941e-21 * f ** 14)
    )
