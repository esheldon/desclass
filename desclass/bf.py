import numpy as np


SLOPE = 2.0
OFF = 16


def predict(*, rmag, amp):
    """
    predict the brighter fatter concentration offset as a function of r band
    magnitude

    Parameters
    ----------
    rmag: float or array
        r band psf magnitude
    amp: float
        Amplitude of relation amp (rmag - OFF)**SLOPE

    Returns
    -------
    concentration offset
    """
    bf_conc = amp*(rmag - OFF)**SLOPE
    try:
        _ = len(bf_conc)
        w, = np.where(rmag < OFF)
        bf_conc[w] = 0.0

        w, = np.where(rmag > 25)
        bf_conc[w] = amp*(25 - OFF)**SLOPE

    except TypeError:
        if rmag < OFF:
            bf_conc = 0.0
        elif rmag > 25:
            bf_conc = amp*(25 - OFF)**SLOPE

    return bf_conc


def get_amp(*, rmag, conc):
    """
    calculate the amplitude assuming the input rmag and concentration follow
    the expected power law

    Parameters
    ----------
    rmag: array
        Array of rmag values, e.g. mean r mag in bins
    conc: array
        Array of conc values, e.g. mean conc in mag bins

    Returns
    -------
    amp: float
        amplitude of (rmag - OFF)**SLOPE
    """
    pv = predict(rmag=rmag, amp=1.0)
    amp = (conc/pv).mean()

    return amp
