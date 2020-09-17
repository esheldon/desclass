import numpy as np
import ngmix
from ngmix.fitting import run_leastsq


def fit_gauss_am(*, rng, obs):
    """
    fit a gaussian to the Observation using adaptive moments

    Parameters
    ----------
    rng: np.RandomState
        The random number generator
    obs: ngmix.Observation
        Observation to fit

    Returns
    -------
    The Admom fitter
    """
    Tguess = ngmix.moments.fwhm_to_T(0.9)
    Tguess *= rng.uniform(low=0.95, high=1.05)

    runner = ngmix.bootstrap.AMRunner(
        obs,
        Tguess,
        rng=rng,
    )
    runner.go(ntry=2)
    return runner.get_fitter()


def erf_func(pars, x, type='falling'):
    from scipy.special import erf

    minval = pars[0]
    maxval = pars[1]
    midval = pars[2]
    scale = pars[3]

    if type == 'falling':
        args = (midval - x)/scale
    else:
        args = (x - midval)/scale

    return minval + 0.5*(maxval - minval)*(1 + erf(args))


def fit_erf(x, y, guess, type='falling'):

    def func(pars):
        return erf_func(pars, x, type)

    def loss(pars):
        model = func(pars)

        return (model - y)

    return run_leastsq(
        loss,
        np.array(guess),
        0,
    )
