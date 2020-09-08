import ngmix


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
