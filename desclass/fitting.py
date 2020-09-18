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


def exp_func(pars, x):
    amp = pars[0]
    off = pars[1]
    sigma = pars[2]

    xc = np.array(x).clip(min=off)

    arg = (xc - off)/sigma
    return amp * (np.exp(arg) - 1)
    # model = np.zeros(x.size)
    # w, = np.where(x > off)
    # if w.size > 0:
    #     arg = (x[w] - off)/sigma
    #     model[w] = amp * (np.exp(arg) - 1)
    #
    # return model


def fit_exp(x, y, guess):
    def loss(pars):
        model = exp_func(pars, x)
        return (model - y)

    return run_leastsq(
        loss,
        np.array(guess),
        0,
        maxfev=4000,
        xtol=1.0e-5,
        ftol=1.0e-5,
    )
